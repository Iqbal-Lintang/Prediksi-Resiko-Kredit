# threshold_classifier.py
class ThresholdClassifier:
    def __init__(self, base_classifier, threshold=0.5):
        self.base_classifier = base_classifier
        self.threshold = threshold
        
    def predict(self, X):
        # Use the calibrated threshold for predictions
        y_proba = self.base_classifier.predict_proba(X)[:, 1]
        return (y_proba >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        # Pass through the probability predictions
        return self.base_classifier.predict_proba(X)