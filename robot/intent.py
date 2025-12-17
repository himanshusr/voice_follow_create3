#!/usr/bin/env python3
"""
Intent parsing for the Create3 voice robot, using Gemma via google-generativeai.

We convert free-form user utterances into a small intent schema:

{
  "intent": "movement_sequence" | "describe_surroundings" | "stop" | "shutdown" | "unknown",
  "actions": [
    {
      "type": "move",
      "direction": "forward" | "backward",
      "distance_m": 0.3,
      "speed": 0.2
    },
    {
      "type": "turn",
      "direction": "left" | "right",
      "degrees": 90
    },
    {
      "type": "stop"
    }
  ],
  "raw": "<original text>"
}
"""

import os
import json
import re
from typing import Any, Dict, List

import google.generativeai as genai


# ========= CONFIG =========

# Reuse the same API key you already use for Gemini vision.
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# Gemma model name for intent parsing.
# Set GEMMA_MODEL_NAME in your env if you want a specific one, e.g.:
#   export GEMMA_MODEL_NAME="gemma-2-2b-it"
GEMMA_MODEL_NAME = os.environ.get("GEMMA_MODEL_NAME", "gemma-3-1b-it")

_MODEL = None  # lazy init


# ========= PROMPT =========

SYSTEM_INSTRUCTIONS = """
You are an intent parser for a small mobile robot.

Your job is to convert a short spoken command into a JSON object that
describes what the robot should do.

SUPPORTED INTENTS:

1) "movement_sequence"
   - User wants the robot to move and/or turn.
   - Return:
     {
       "intent": "movement_sequence",
       "actions": [
         {
           "type": "move",
           "direction": "forward" or "backward",
           "distance_m": <number in meters>,
           "speed": <optional number in m/s>
         },
         {
           "type": "turn",
           "direction": "left" or "right",
           "degrees": <number>
         },
         {
           "type": "stop"
         }
       ]
     }

   - Distances:
       "one meter" -> 1.0
       "half a meter" or "0.5 meters" -> 0.5
       "thirty centimeters" -> 0.3
   - If the user does not specify a distance, omit the "distance_m" field.

2) "describe_surroundings"
   - User wants the robot to describe what it sees with its camera.
   - Examples:
       "What do you see?"
       "Describe your surroundings."
       "Tell me what is around you."
   - Return:
       { "intent": "describe_surroundings" }

3) "stop"
   - User wants the robot to stop immediately.
   - Examples:
       "Stop", "Emergency stop", "Freeze"
   - Return:
       { "intent": "stop" }

4) "shutdown"
   - User wants to turn off the voice interface / program.
   - Examples:
       "shut down", "power off", "exit program"
   - Return:
       { "intent": "shutdown" }

5) "unknown"
   - Anything else that does not clearly match the above.
   - Return:
       { "intent": "unknown" }

REQUIREMENTS:

- ALWAYS respond with PURE JSON. Do NOT include explanations.
- Do NOT wrap the JSON in backticks.
- If you return "movement_sequence", you MUST include an "actions" array.
- If multiple actions are requested in one sentence, list them in order.
  For example:
    "Go forward one meter then turn left ninety degrees"
  could map to:
    {
      "intent": "movement_sequence",
      "actions": [
        {
          "type": "move",
          "direction": "forward",
          "distance_m": 1.0
        },
        {
          "type": "turn",
          "direction": "left",
          "degrees": 90
        }
      ]
    }
"""


def _get_model():
    """Lazily initialize Gemma model."""
    global _MODEL
    if _MODEL is None:
        if not API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set; cannot use Gemma intent parser.")
        genai.configure(api_key=API_KEY)
        _MODEL = genai.GenerativeModel(GEMMA_MODEL_NAME)
    return _MODEL


def _extract_json(text: str) -> str:
    """
    Extract the first JSON object from the model's response text.
    Handles cases like:
      { ... }
      ```json
      { ... }
      ```
    """
    if not text:
        raise ValueError("Empty response text.")

    # Look for fenced ```json blocks
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Otherwise, find the first {...} in the output
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    # If nothing looks like JSON, return raw (will fail JSON parsing)
    return text.strip()


def _normalize_result(data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """
    Ensure the result dict has at least:
      - "intent" (string)
      - "actions" (list) for movement_sequence
      - "raw" (original text)
    """
    intent = str(data.get("intent", "unknown")).strip() or "unknown"

    if intent == "movement_sequence":
        actions = data.get("actions") or []
        if not isinstance(actions, list):
            actions = []
        # Light cleanup: ensure each action has a 'type'
        cleaned: List[Dict[str, Any]] = []
        for a in actions:
            if isinstance(a, dict) and a.get("type"):
                cleaned.append(a)
        data["actions"] = cleaned
    else:
        data["actions"] = []

    data["intent"] = intent
    data["raw"] = raw_text
    return data


def _fallback_keyword_parser(text: str) -> Dict[str, Any]:
    """
    Very simple backup parser if Gemma is unavailable.
    """
    lower = text.lower()

    if any(w in lower for w in ["stop", "emergency stop", "freeze"]):
        return {"intent": "stop", "actions": [], "raw": text}

    if any(w in lower for w in ["shut down", "shutdown", "power off", "exit program"]):
        return {"intent": "shutdown", "actions": [], "raw": text}

    if any(w in lower for w in ["what do you see", "surroundings", "around you", "describe what you see"]):
        return {"intent": "describe_surroundings", "actions": [], "raw": text}

    if "forward" in lower:
        return {
            "intent": "movement_sequence",
            "actions": [
                {"type": "move", "direction": "forward", "distance_m": 0.3}
            ],
            "raw": text,
        }

    return {"intent": "unknown", "actions": [], "raw": text}


def parse_intent(text: str) -> Dict[str, Any]:
    """
    Public entry point used by wake_voice_teleop.py.

    - Tries Gemma first.
    - On any error (API, JSON), falls back to keyword parsing.
    """
    text = (text or "").strip()
    if not text:
        return {"intent": "unknown", "actions": [], "raw": text}

    try:
        model = _get_model()
        prompt = SYSTEM_INSTRUCTIONS + "\n\nUser command:\n" + text + "\n\nJSON:"
        response = model.generate_content(prompt)
        raw_out = (response.text or "").strip()
        json_str = _extract_json(raw_out)
        data = json.loads(json_str)
        return _normalize_result(data, text)
    except Exception as e:
        print(f"[INTENT] Gemma parsing failed, falling back to keywords: {e}")
        return _fallback_keyword_parser(text)
