# run_button.py (non-interactive version)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def run_evaluation_and_reconstruction(model_path, X_test_scaled, y_test_cat, class_names):
    """
    Evaluates a trained Keras model and visualizes reconstruction results:
      - prints classification report and accuracy
      - shows confusion matrix
      - visualizes data reconstruction quality
    Parameters:
        model_path : str
            Path to the .keras Keras model file
        X_test_scaled : np.ndarray
            Scaled test features
        y_test_cat : np.ndarray
            One-hot encoded test labels
        class_names : list of str
            Class label names in the correct order
    """
    try:
        # Load model with custom objects to handle 'mse' function
        model = load_model(model_path, custom_objects={
            'mse': MeanSquaredError(),
            'MeanSquaredError': MeanSquaredError
        })
        print(f"✅ Loaded model from: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    try:
        predictions = model.predict(X_test_scaled)
        # For multi-output model, get both outputs
        if isinstance(predictions, dict):
            reconstructed_data = predictions['reconstructed_input']
            pred_probs = predictions['class_probability']
        else:
            reconstructed_data = predictions[0]
            pred_probs = predictions[1]
        
        y_pred = np.argmax(pred_probs, axis=1)
        y_true = np.argmax(y_test_cat, axis=1)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return

    # === 诊断分类评估 ===
    print("\n📊 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"✅ Overall Accuracy: {acc:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # === 数据重构结果可视化 ===
    # Calculate reconstruction error
    reconstruction_error = np.mean(np.square(X_test_scaled - reconstructed_data), axis=1)
    
    plt.subplot(1, 2, 2)
    # Reconstruction error by class
    colors = ['blue', 'orange', 'green', 'red']
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_errors = reconstruction_error[class_mask]
        plt.scatter(np.ones(len(class_errors)) * i, class_errors, 
                   alpha=0.6, c=colors[i % len(colors)], label=class_name)
    
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Error by Class')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === 重构质量统计 ===
    print("\n🔧 Reconstruction Quality Analysis:")
    print(f"Overall reconstruction MSE: {np.mean(reconstruction_error):.6f}")
    print(f"Reconstruction error std: {np.std(reconstruction_error):.6f}")
    
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_errors = reconstruction_error[class_mask]
        if len(class_errors) > 0:
            print(f"{class_name}: MSE = {np.mean(class_errors):.6f} ± {np.std(class_errors):.6f}")

    # === 样本重构可视化 ===
    # Show original vs reconstructed for first few samples
    print("\n📈 Sample Reconstruction Visualization:")
    n_samples = min(5, len(X_test_scaled))
    
    plt.figure(figsize=(15, 4))
    for i in range(n_samples):
        # Original data
        plt.subplot(2, n_samples, i + 1)
        plt.plot(X_test_scaled[i], 'b-', label='Original')
        plt.title(f'Sample {i+1}\nClass: {class_names[y_true[i]]}')
        plt.ylabel('Original')
        if i == 0:
            plt.legend()
        
        # Reconstructed data
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.plot(reconstructed_data[i], 'r-', label='Reconstructed')
        plt.ylabel('Reconstructed')
        plt.xlabel('Feature Index')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Detailed accuracy explanation
    print("\n📈 Detailed Classification Analysis:")
    total_samples = len(y_true)
    print(f"Total test samples: {total_samples}")
    
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_total = np.sum(class_mask)
        class_correct = np.sum((y_true == i) & (y_pred == i))
        class_accuracy = (class_correct / class_total * 100) if class_total > 0 else 0
        print(f"{class_name}: {class_correct}/{class_total} correct ({class_accuracy:.2f}%)")
