import mediapipe as mp
print("version:", mp.__version__)
print("attrs:", [a for a in dir(mp) if not a.startswith('_')])
print("has solutions:", hasattr(mp, 'solutions'))
print("has tasks:", hasattr(mp, 'tasks'))