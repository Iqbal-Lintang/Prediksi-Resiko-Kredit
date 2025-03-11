# calibration_script.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score
from threshold_classifier import ThresholdClassifier

# Load the original saved model
original_model = joblib.load('best_risk_prediction_model.pkl')

# Create the calibrated model
calibrated_model = ThresholdClassifier(original_model, threshold=0.3)

# Save the calibrated model
joblib.dump(calibrated_model, 'calibrated_risk_prediction_model.pkl')
print("Created calibrated model with threshold=0.3, saved to calibrated_risk_prediction_model.pkl")