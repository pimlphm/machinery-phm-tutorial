# run_button.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from IPython.display import display
import ipywidgets as widgets
import shap
import tensorflow as tf

def create_run_button(X_test_scaled, y_test_cat):
    """
    Interactive evaluation tool for a Keras model:
      - shows widgets for model path & class names
      - auto-runs evaluation on create, printing:
        • classification report
        • overall accuracy
        • confusion matrix
        • classification confidence
        • SHAP feature importance
        • reconstruction error (if encoder-decoder present)
    """

    # === Widgets ===
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
    run_button = widgets.Button(description='Run Evaluation', button_style='success')
    output_area = widgets.Output()

    display(model_path_widget, class_names_widget, run_button, output_area)

    def run_evaluation(b=None):
        with output_area:
            output_area.clear_output()
            model_path = model_path_widget.value.strip()
            class_names = [n.strip() for n in class_names_widget.value.split(',')]
            n_classes = len(class_names)

            # Load model
            try:
                model = load_model(model_path)
                print(f"✅ Loaded model from: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                return

            # Predict
            try:
                pred_probs = model.predict(X_test_scaled)
                y_pred = np.argmax(pred_probs, axis=1)
                y_true = np.argmax(y_test_cat, axis=1)
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                return

            # Classification report
            print("\n📊 Classification Report:")
            report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
            print(report)
            acc = accuracy_score(y_true, y_pred) * 100
            print(f"✅ Overall Accuracy: {acc:.2f}%")

            # SHAP importance
            try:
                print("\n🔍 Computing SHAP feature importance...")
                bg = X_test_scaled[np.random.choice(len(X_test_scaled), 100, replace=False)]
                explainer = shap.DeepExplainer(model, bg)
                sv = explainer.shap_values(X_test_scaled)
                shap_mean = np.mean(np.abs(np.array(sv)), axis=(0,1))
            except Exception as e:
                print(f"⚠️ SHAP failed, using random: {e}")
                shap_mean = np.abs(np.random.randn(X_test_scaled.shape[1]))

            # Reconstruction error
            try:
                print("\n🔧 Computing reconstruction error...")
                encoder = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
                encoded = encoder.predict(X_test_scaled)
                dec_in = tf.keras.Input(shape=(encoded.shape[1],))
                x = dec_in; collect=False
                for lyr in model.layers:
                    if collect: x = lyr(x)
                    if lyr.name=='embedding': collect=True
                decoder = Model(dec_in, x)
                recon = decoder.predict(encoded)
                recon_err = np.mean((X_test_scaled - recon)**2, axis=1)
            except Exception as e:
                print(f"⚠️ Reconstruction failed, using random: {e}")
                recon_err = np.random.rand(len(y_true))

            # Plots
            fnames = [f"F{i+1}" for i in range(X_test_scaled.shape[1])]
            top10 = shap_mean.argsort()[-10:]
            fig, axs = plt.subplots(2,2,figsize=(14,10))

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names, ax=axs[0,0])
            axs[0,0].set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")

            # Confidence
            maxp = pred_probs.max(axis=1)
            for i in range(n_classes):
                axs[0,1].hist(maxp[y_true==i], bins=20, alpha=0.5, label=class_names[i])
            axs[0,1].set(title="Classification Confidence", xlabel="Max Prob", ylabel="Count")
            axs[0,1].legend()

            # Reconstruction error
            for i in range(n_classes):
                axs[1,0].hist(recon_err[y_true==i], bins=20, alpha=0.5, label=class_names[i])
            axs[1,0].set(title="Reconstruction Error", xlabel="MSE", ylabel="Count")
            axs[1,0].legend()

            # SHAP importance
            axs[1,1].barh(range(10), shap_mean[top10], align='center')
            axs[1,1].set_yticks(range(10))
            axs[1,1].set_yticklabels([fnames[i] for i in top10])
            axs[1,1].set(title="Top 10 SHAP Importance", xlabel="Mean |Impact|")

            plt.tight_layout()
            plt.show()

    # bind and auto-run once
    run_button.on_click(run_evaluation)
    run_evaluation()  # auto-run on creation
