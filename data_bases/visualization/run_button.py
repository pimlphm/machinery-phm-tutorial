# run_button.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from IPython.display import display
import ipywidgets as widgets

def create_run_button(X_test_scaled, y_test_cat):
    """
    Create an interactive widget to load a trained model,
    evaluate predictions on a test set, and visualize:
    - Confusion matrix
    - Confidence distribution
    - Reconstruction error (simulated)
    - Feature importance (simulated)

    Parameters:
        X_test_scaled (ndarray): Preprocessed test features
        y_test_cat (ndarray): One-hot encoded test labels
    """

    # === Step 1: UI widgets ===
    model_path_widget = widgets.Text(
        value='best_model_diagnostics.h5',
        description='Model path:',
        layout=widgets.Layout(width='500px')
    )

    class_names_widget = widgets.Text(
        value='Normal,Misalignment,Unbalance,Looseness',
        description='Class names:',
        layout=widgets.Layout(width='500px')
    )

    run_button = widgets.Button(
        description='Run Evaluation',
        button_style='success'
    )

    display(model_path_widget, class_names_widget, run_button)

    # === Step 2: Evaluation function ===
    def run_evaluation(b):
        model_path = model_path_widget.value.strip()
        class_names = [name.strip() for name in class_names_widget.value.split(',')]
        n_classes = len(class_names)

        try:
            model = load_model(model_path)
            print(f"✅ Loaded model from: {model_path}")
        except:
            print(f"❌ Failed to load model: {model_path}")
            return

        try:
            pred_probs = model.predict(X_test_scaled)
            y_pred = np.argmax(pred_probs, axis=1)
            y_true = np.argmax(y_test_cat, axis=1)
        except:
            print("❌ Prediction failed. Make sure inputs are valid arrays.")
            return

        # Simulated data (you may replace with real values)
        reconstruction_error = np.random.rand(len(y_true))
        input_dim = X_test_scaled.shape[1]
        importance = np.abs(np.random.randn(input_dim))
        top_idx = importance.argsort()[-10:]
        feature_names = [f"F{i+1}" for i in range(input_dim)]

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axs[0, 0])
        axs[0, 0].set_title("Confusion Matrix")
        axs[0, 0].set_xlabel("Predicted")
        axs[0, 0].set_ylabel("Actual")

        # 2. Confidence
        max_probs = pred_probs.max(axis=1)
        for i in range(n_classes):
            axs[0, 1].hist(max_probs[y_true == i], bins=20, alpha=0.5, label=class_names[i])
        axs[0, 1].set_title("Classification Confidence by Class")
        axs[0, 1].set_xlabel("Max Probability")
        axs[0, 1].set_ylabel("Count")
        axs[0, 1].legend()

        # 3. Reconstruction error
        for i in range(n_classes):
            axs[1, 0].hist(reconstruction_error[y_true == i], bins=20, alpha=0.5, label=class_names[i])
        axs[1, 0].set_title("Reconstruction Error by Class (Simulated)")
        axs[1, 0].set_xlabel("Reconstruction Error")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].legend()

        # 4. Feature importance
        axs[1, 1].barh(range(10), importance[top_idx], align='center')
        axs[1, 1].set_yticks(range(10))
        axs[1, 1].set_yticklabels([feature_names[i] for i in top_idx])
        axs[1, 1].set_title("Top 10 Feature Importance (Simulated)")
        axs[1, 1].set_xlabel("Importance Score")

        plt.tight_layout()
        plt.show()

    run_button.on_click(run_evaluation)
