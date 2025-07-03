# Import required packages
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def interactive_layer_pca_visualization(model, X_test_scaled, y_test):
    """
    Create an interactive slider to visualize PCA results from different layers of the model
    
    Args:
        model: The trained diagnostic model
        X_test_scaled: Scaled test data
        y_test: Test labels
    """
    
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
