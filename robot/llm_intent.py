import os
import json
import textwrap
from typing import Any, Dict, Optional

import requests


class LLMIntentError(Exception):
    """Raised when the LLM intent call fails or returns bad data."""


class LLMIntentParser:
    """
    Intent parser that calls a local/remote LLM (e.g. Gemma via Ollama)
    and returns a normalized intent dict for the robot.

    It expects an Ollama-compatible HTTP API:
      POST /api/chat
      {
        "model": "gemma3:4b",
        "messages": [{"role": "user", "content": "..."}],
        "stream": false
      }
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_INTENT_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("LLM_INTENT_MODEL", "gemma3:4b")
        self.temperature = float(os.getenv("LLM_INTENT_TEMPERATURE", str(temperature or 0.0)))

        print(f"[NLU] Using Ollama model for intent: {self.model} at {self.base_url}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, user_text: str) -> Dict[str, Any]:
        """
        Main entrypoint. Returns a dict like:
          {
            "intent": "movement_sequence" | "stop" | "describe_surroundings" | "shutdown" | "unknown",
            "actions": [...],
            "raw": "<original text>"
          }
        """
        if not user_text or not user_text.strip():
            return {"intent": "unknown", "actions": [], "raw": user_text}

        prompt = self._build_prompt(user_text)
        raw_reply = self._call_ollama_chat(prompt)
        intent = self._extract_json(raw_reply)

        if not isinstance(intent, dict):
            raise LLMIntentError(f"LLM did not return a JSON object: {intent!r}")

        # Normalize and ensure required keys.
        intent.setdefault("intent", "unknown")
        intent.setdefault("actions", [])
        intent.setdefault("raw", user_text)

        # Basic sanity: enforce expected structure for movement_sequence
        if intent["intent"] == "movement_sequence":
            actions = intent.get("actions", [])
            if not isinstance(actions, list):
                intent["actions"] = []
            else:
                normalized = []
                for a in actions:
                    if not isinstance(a, dict):
                        continue
                    atype = a.get("type")
                    if atype not in ("move", "turn", "stop"):
                        continue
                    normalized.append(a)
                intent["actions"] = normalized

        return intent

    # ------------------------------------------------------------------
    # Prompt + HTTP helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, user_text: str) -> str:
        """
        Build a strict JSON-only prompt for the LLM.
        """
        return textwrap.dedent(f"""
        You are the intent parser for a small mobile robot (iRobot Create3).
        Your job is to read a human's voice command and output a single JSON object
        describing what the robot should do.

        The JSON must follow this schema:

        {{
          "intent": "<one of: movement_sequence, stop, describe_surroundings, shutdown, unknown>",
          "actions": [
            {{
              "type": "move",
              "direction": "forward" | "backward",
              "distance_m": <float>,
              "speed": <float OPTIONAL>
            }},
            {{
              "type": "turn",
              "direction": "left" | "right",
              "degrees": <float>
            }},
            {{
              "type": "stop"
            }}
          ],
          "raw": "<copy of the original user command>"
        }}

        Rules:
        - If the user wants the robot to move or turn, use intent "movement_sequence"
          and fill in a list of actions in the right order.
        - Distances should be in meters; if the user says "feet", convert to meters
          (1 foot ≈ 0.3048 meters).
        - Angles should be in degrees.
        - If the user says to stop, use intent "stop".
        - If the user says "describe surroundings", "what do you see", etc.,
          use intent "describe_surroundings".
        - If the user says "shut down", "exit", "power off voice", etc.,
          use intent "shutdown".
        - If you cannot understand the command, use intent "unknown".

        IMPORTANT:
        - Output MUST be valid JSON.
        - Do NOT include any explanation, comments, or extra text; only the JSON.
        - Make sure the JSON is minified or pretty-printed, but valid.

        USER COMMAND:
        "{user_text.strip()}"
        """)

    def _call_ollama_chat(self, prompt: str) -> str:
        """
        Call the Ollama /api/chat endpoint and return the model's text content.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
        except Exception as e:
            raise LLMIntentError(f"HTTP error calling LLM: {e}") from e

        if resp.status_code != 200:
            raise LLMIntentError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except Exception as e:
            raise LLMIntentError(f"Invalid JSON response from LLM: {e}") from e

        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise LLMIntentError(f"LLM response missing 'message.content': {data}")

        return content

    def _extract_json(self, text: str) -> Any:
        """
        LLM might wrap JSON in other text. Extract the first {...} block
        and parse it.
        """
        text = text.strip()
        if not text:
            raise LLMIntentError("Empty response from LLM.")

        # If it already looks like pure JSON, try directly.
        if text[0] == "{" and text[-1] == "}":
            try:
                return json.loads(text)
            except Exception:
                pass

        # Otherwise, find the first '{' and last '}' and parse that substring.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMIntentError(f"Could not find JSON object in: {text!r}")

        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception as e:
            raise LLMIntentError(f"Failed to parse JSON from LLM text: {e}; snippet={snippet!r}") from e
