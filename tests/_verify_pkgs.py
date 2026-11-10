import cv2, numpy as np, mediapipe as mp, gradio as gr, requests, edge_tts
print("cv2:", cv2.__version__)
print("numpy:", np.__version__)
print("mediapipe:", mp.__version__, "| has solutions:", hasattr(mp, 'solutions'))
print("gradio:", gr.__version__)
pose = mp.solutions.pose.Pose(static_image_mode=True)
hands = mp.solutions.hands.Hands(static_image_mode=False)
print("Pose OK, Hands OK")
print("ALL IMPORTS OK")