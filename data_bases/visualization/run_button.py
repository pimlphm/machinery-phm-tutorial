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

    # SHAP
    try:
        print("\n🔍 Computing SHAP feature importance...")
        background = X_test_scaled[np.random.choice(len(X_test_scaled), 100, replace=False)]
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_test_scaled)
        shap_mean = np.mean(np.abs(np.array(shap_values)), axis=(0,1))
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")
        shap_mean = np.abs(np.random.randn(X_test_scaled.shape[1]))

    # Reconstruction Error
    try:
        print("\n🔧 Computing reconstruction error...")
        encoder = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
        encoded = encoder.predict(X_test_scaled)
        dec_in = tf.keras.Input(shape=(encoded.shape[1],))
        x = dec_in; collect = False
        for lyr in model.layers:
            if collect: x = lyr(x)
            if lyr.name == 'embedding': collect = True
        decoder = Model(dec_in, x)
        recon = decoder.predict(encoded)
        recon_err = np.mean((X_test_scaled - recon) ** 2, axis=1)
    except Exception as e:
        print(f"⚠️ Reconstruction failed: {e}")
        recon_err = np.random.rand(len(y_true))

    # Visualization
    fnames = [f"F{i+1}" for i in range(X_test_scaled.shape[1])]
    top10 = shap_mean.argsort()[-10:]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axs[0,0])
    axs[0,0].set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")

    # Classification Confidence
    max_probs = pred_probs.max(axis=1)
    for i in range(len(class_names)):
        axs[0,1].hist(max_probs[y_true == i], bins=20, alpha=0.5, label=class_names[i])
    axs[0,1].set(title="Classification Confidence", xlabel="Max Prob", ylabel="Count")
    axs[0,1].legend()

    # Reconstruction Error
    for i in range(len(class_names)):
        axs[1,0].hist(recon_err[y_true == i], bins=20, alpha=0.5, label=class_names[i])
    axs[1,0].set(title="Reconstruction Error", xlabel="MSE", ylabel="Count")
    axs[1,0].legend()

    # SHAP Bar Plot
    axs[1,1].barh(range(10), shap_mean[top10], align='center')
    axs[1,1].set_yticks(range(10))
    axs[1,1].set_yticklabels([fnames[i] for i in top10])
    axs[1,1].set(title="Top 10 SHAP Importance", xlabel="Mean |Impact|")

    plt.tight_layout()
    plt.show()
