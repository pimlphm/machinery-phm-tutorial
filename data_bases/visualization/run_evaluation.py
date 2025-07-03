# run_button.py (non-interactive version)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def run_evaluation(model_path, X_test_scaled, y_test_cat, class_names):
    """
    Evaluates a trained Keras model:
      - prints classification report and accuracy
      - shows confusion matrix
    Parameters:
        model_path : str
            Path to the .h5 Keras model file
        X_test_scaled : np.ndarray
            Scaled test features
        y_test_cat : np.ndarray
            One-hot encoded test labels
        class_names : list of str
            Class label names in the correct order
    """
    try:
        model = load_model(model_path)
        print(f"✅ Loaded model from: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    try:
        pred_probs = model.predict(X_test_scaled)
        y_pred = np.argmax(pred_probs, axis=1)
        y_true = np.argmax(y_test_cat, axis=1)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return

    # Classification metrics
    print("\n📊 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"✅ Overall Accuracy: {acc:.2f}%")

    # Visualization - Confusion Matrix only
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Detailed accuracy explanation
    print("\n📈 Detailed Accuracy Analysis:")
    total_samples = len(y_true)
    print(f"Total test samples: {total_samples}")
    
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_total = np.sum(class_mask)
        class_correct = np.sum((y_true == i) & (y_pred == i))
        class_accuracy = (class_correct / class_total * 100) if class_total > 0 else 0
        print(f"{class_name}: {class_correct}/{class_total} correct ({class_accuracy:.2f}%)")
