# Handwriting Recognition with MNIST + Mediapipe

Link to download model: https://drive.google.com/file/d/1mqCcwghY2ek0KRlyNnrju0UjPWlubvS7/view?usp=sharing

This project combines a custom-trained MNIST neural network with Mediapipe hand tracking.  
You can ✋ use your hand as a pen to draw digits in the air, and the model will predict the digit in real-time.  

---

## Features
- Model trained on mnist 
- Detects ✋ open palm (stop) and 👆 index finger (draw).
- Converts finger drawings into 28×28 MNIST-style images for prediction.
- Shows predicted digit live on the video feed.

---
## Instructions to use live recognition
- Pinch your index finger to start drawing.
- Open your palm to stop and predict the digit.
- Press ESC to exit.
