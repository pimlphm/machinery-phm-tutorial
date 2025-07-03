# Import required packages
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def interactive_layer_pca_visualization(model_path, X_test_scaled, y_test):
    """
    Create an interactive slider to visualize PCA results from different layers of the model
    
    Args:
        model_path: Path to the trained diagnostic model file
        X_test_scaled: Scaled test data
        y_test: Test labels
    """
    
    # Load the model from the provided path
    model = load_model(model_path)
    
    def visualize_features_pca(features, labels, n_components=2, label_names=None, colors=None, title="PCA Visualization"):
        """
        Helper function to visualize PCA results
        """
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_scaled)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        if label_names is None:
            label_names = [f'Class {i}' for i in unique_labels]
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                       c=colors[i], label=label_names[i], alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def update_pca_plot(layer_index):
        # Clear previous plot
        plt.clf()
        
        # Define latent feature extractor model up to selected layer
        latent_model = Model(inputs=model.input, outputs=model.get_layer(index=layer_index).output)
        
        # Extract latent features from test set
        X_test_latent = latent_model.predict(X_test_scaled, verbose=0)
        
        # Visualize PCA results
        visualize_features_pca(
            X_test_latent, y_test,
            n_components=2,
            label_names=['Normal', 'Misalignment', 'Unbalance', 'Looseness'],
            colors=['#2E8B57', '#FF6347', '#4169E1', '#FF8C00'],
            title=f'2D PCA of Features from Layer {layer_index} ({model.get_layer(index=layer_index).name})'
        )
        plt.show()
    
    # Create slider widget
    layer_slider = widgets.IntSlider(
        value=11,
        min=0,
        max=len(model.layers) - 1,
        step=1,
        description='Layer Index:',
        style={'description_width': 'initial'},
        continuous_update=False
    )
    
    # Create interactive widget
    interactive_plot = widgets.interactive(update_pca_plot, layer_index=layer_slider)
    
    # Display the widget
    display(interactive_plot)
