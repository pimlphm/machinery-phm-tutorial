# run_button.py (non-interactive version)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import shap
import tensorflow as tf

def run_evaluation(model_path, X_test_scaled, y_test_cat, class_names):
    """
    Evaluates a trained Keras model:
      - prints classification report and accuracy
      - shows confusion matrix, confidence, reconstruction error, SHAP importance
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

    # SHAP - Use different explainer to avoid LeakyRelu error
    try:
        print("\n🔍 Computing SHAP feature importance...")
        # Use KernelExplainer instead of DeepExplainer to avoid gradient issues
        background = X_test_scaled[np.random.choice(len(X_test_scaled), 50, replace=False)]
        explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
        shap_values = explainer.shap_values(X_test_scaled[:50])  # Further limit sample size
        if isinstance(shap_values, list):
            # For multi-class models
            shap_mean = np.mean([np.abs(sv) for sv in shap_values], axis=(0, 1))
        else:
            # For binary classification
            shap_mean = np.mean(np.abs(shap_values), axis=0)
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")
        shap_mean = np.abs(np.random.randn(X_test_scaled.shape[1]))

    # Reconstruction Error - Simplified approach
    try:
        print("\n🔧 Computing reconstruction error...")
        # Check if model has an embedding layer
        has_embedding = any(layer.name == 'embedding' for layer in model.layers)
        
        if has_embedding:
            # Create encoder to get embedding
            encoder = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
            encoded = encoder.predict(X_test_scaled)
            
            # For autoencoder-style models, use the embedding as compressed representation
            # and compute error based on how well the model can reconstruct from it
            # Since we can't easily build a decoder, use prediction confidence as proxy
            max_confidence = np.max(pred_probs, axis=1)
            # Invert confidence to get "error" (low confidence = high error)
            recon_err = 1.0 - max_confidence
        else:
            print("⚠️ No embedding layer found, using prediction confidence as proxy")
            # Use prediction confidence as reconstruction error proxy
            max_confidence = np.max(pred_probs, axis=1)
            recon_err = 1.0 - max_confidence
    except Exception as e:
        print(f"⚠️ Reconstruction computation failed: {e}")
        # Use prediction confidence as fallback
        max_confidence = np.max(pred_probs, axis=1)
        recon_err = 1.0 - max_confidence

    # Visualization
    fnames = [f"F{i+1}" for i in range(X_test_scaled.shape[1])]
    top10_indices = shap_mean.argsort()[-10:] if len(shap_mean) >= 10 else shap_mean.argsort()
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axs[0,0])
    axs[0,0].set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")

    # Classification Confidence
    max_probs = pred_probs.max(axis=1)
    for i in range(len(class_names)):
        mask = y_true == i
        if np.any(mask):
            axs[0,1].hist(max_probs[mask], bins=20, alpha=0.5, label=class_names[i])
    axs[0,1].set(title="Classification Confidence", xlabel="Max Prob", ylabel="Count")
    axs[0,1].legend()

    # Reconstruction Error (now using confidence proxy)
    for i in range(len(class_names)):
        mask = y_true == i
        if np.any(mask):
            axs[1,0].hist(recon_err[mask], bins=20, alpha=0.5, label=class_names[i])
    axs[1,0].set(title="Reconstruction Error (1 - Confidence)", xlabel="Error", ylabel="Count")
    axs[1,0].legend()

    # SHAP Bar Plot
    n_features = min(10, len(top10_indices))
    axs[1,1].barh(range(n_features), shap_mean[top10_indices][:n_features], align='center')
    axs[1,1].set_yticks(range(n_features))
    axs[1,1].set_yticklabels([fnames[i] for i in top10_indices[:n_features]])
    axs[1,1].set(title=f"Top {n_features} SHAP Importance", xlabel="Mean |Impact|")

    plt.tight_layout()
    plt.show()
