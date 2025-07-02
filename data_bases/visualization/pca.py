import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
 
def visualize_features_pca(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    label_names: list[str] | None = None,
    colors: list[str] | None = None,
    title: str = 'PCA Feature Projection'
):
    """
    Projects high-dimensional features X and labels y into PCA space,
    then plots the first 2 or 3 principal components as a scatter plot.

    Parameters:
        X : np.ndarray, shape (N, D)
            Feature matrix, N samples × D features.
        y : np.ndarray, shape (N,)
            Integer class labels (0,1,2,...).
        n_components : int, default=2
            Number of principal components to compute (must be 2 or 3).
        label_names : list of str, optional
            Human-readable names for each integer class.
            If None, classes are labeled by their integer value.
        colors : list of color strings, optional
            One color per class. If None, uses default cycle.
        title : str, default='PCA Feature Projection'
            Plot title.
    """

    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for plotting.")

    # Default label names = stringified class indices
    classes = np.unique(y)
    if label_names is None:
        label_names = [str(c) for c in classes]
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(classes))]

    # Compute PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Plot
    if n_components == 2:
        plt.figure(figsize=(10,7))
        for cls, name, col in zip(classes, label_names, colors):
            mask = (y == cls)
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                label=name,
                color=col,
                s=40,
                alpha=0.7
            )
        plt.xlabel('PC 1', fontsize=14)
        plt.ylabel('PC 2', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(title='Classes')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    else:  # 3D
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        for cls, name, col in zip(classes, label_names, colors):
            mask = (y == cls)
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                X_pca[mask, 2],
                label=name,
                color=col,
                s=40,
                alpha=0.7
            )
        ax.set_xlabel('PC 1', fontsize=12)
        ax.set_ylabel('PC 2', fontsize=12)
        ax.set_zlabel('PC 3', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(title='Classes')
        plt.tight_layout()
        plt.show()
