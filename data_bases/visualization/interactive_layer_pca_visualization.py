import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import MeanSquaredError
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def interactive_layer_pca_visualization(model_path, X_test_scaled, y_test, default_normalize=True):
    """
    Create an interactive slider to visualize PCA results from different layers of the model
    
    Args:
        model_path: Path to the trained diagnostic model file
        X_test_scaled: Scaled test data
        y_test: Test labels
        default_normalize: Default value for normalization checkbox
    """
    
    # Load the model from the provided path with custom objects
    custom_objects = {'mse': MeanSquaredError()}
    model = load_model(model_path, custom_objects=custom_objects)
    
    def visualize_features_pca(features, labels, n_components=2, label_names=None, colors=None, title="PCA Visualization"):
        """
        Helper function to visualize PCA results
        
        Args:
            features: Input features (already normalized if needed)
            labels: Labels for the data
            n_components: Number of principal components (2 or 3)
            label_names: Names for the labels
            colors: Colors for different classes
            title: Plot title
        """
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features)
        
        # Create plot
        if n_components == 2:
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
            
        elif n_components == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            unique_labels = np.unique(labels)
            if colors is None:
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            if label_names is None:
                label_names = [f'Class {i}' for i in unique_labels]
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(features_pca[mask, 0], features_pca[mask, 1], features_pca[mask, 2],
                          c=colors[i], label=label_names[i], alpha=0.7, s=50)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
            ax.set_title(title)
            ax.legend()
    
    def update_pca_plot(layer_index, n_components, normalize):
        # Clear previous plot
        plt.clf()
        
        # Define latent feature extractor model up to selected layer
        latent_model = Model(inputs=model.input, outputs=model.get_layer(index=layer_index).output)
        
        # Extract latent features from test set
        X_test_latent = latent_model.predict(X_test_scaled, verbose=0)
        
        # Flatten features if needed
        if len(X_test_latent.shape) > 2:
            X_test_latent = X_test_latent.reshape(X_test_latent.shape[0], -1)
        
        # Apply standardization if requested (before PCA)
        if normalize:
            scaler = StandardScaler()
            X_test_latent = scaler.fit_transform(X_test_latent)
        
        # Prepare title with normalization info
        norm_text = "with normalization" if normalize else "without normalization"
        component_text = "2D" if n_components == 2 else "3D"
        
        # Visualize PCA results
        visualize_features_pca(
            X_test_latent, y_test,
            n_components=n_components,
            label_names=['Normal', 'Misalignment', 'Unbalance', 'Looseness'],
            colors=['#2E8B57', '#FF6347', '#4169E1', '#FF8C00'],
            title=f'{component_text} PCA of Features from Layer {layer_index} ({model.get_layer(index=layer_index).name}) - {norm_text}'
        )
        plt.show()
    
    # Create widgets
    layer_slider = widgets.IntSlider(
        value=11,
        min=0,
        max=len(model.layers) - 1,
        step=1,
        description='Layer Index:',
        style={'description_width': 'initial'},
        continuous_update=False
    )
    
    # Create dropdown for PCA dimensions
    dimension_dropdown = widgets.Dropdown(
        options=[('2D', 2), ('3D', 3)],
        value=2,
        description='PCA Dimensions:',
        style={'description_width': 'initial'}
    )
    
    # Create checkbox for normalization with default value from parameter
    normalize_checkbox = widgets.Checkbox(
        value=default_normalize,
        description='Apply Normalization',
        style={'description_width': 'initial'}
    )
    
    # Create interactive widget
    interactive_plot = widgets.interactive(
        update_pca_plot, 
        layer_index=layer_slider,
        n_components=dimension_dropdown,
        normalize=normalize_checkbox
    )
    
    # Display the widget
    display(interactive_plot)
