import cv2
import pvporcupine
import whisper
import google.generativeai as genai
import rclpy

print("cv2 version:", cv2.__version__)
print("Porcupine version OK")
print("Whisper model list:", whisper.available_models()[:3])
print("google-generativeai OK")
print("rclpy OK")
