import sys
import os

# Add src to path
sys.path.append(os.getcwd())

try:
    from src.predict import predict_tomorrow, predict_price
    print("Import successful")
    
    direction, confidence, probs = predict_tomorrow()
    print(f"Prediction: {direction}, Confidence: {confidence}")
    print("Probs:", probs)
    
    price = predict_price()
    print(f"Predicted Price: {price}")
    
    print("Test passed!")
except Exception as e:
    print(f"Test failed: {e}")
    sys.exit(1)
