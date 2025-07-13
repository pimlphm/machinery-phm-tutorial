from sklearn.model_selection import train_test_split
import numpy as np

def prepare_dataset(X_normal, X_misalign, X_unbalance, X_looseness, test_size=0.15, val_size=0.15, random_state=42):
    """
    Combine features and labels, then split into train/validation/test sets.
    
    Parameters:
    -----------
    X_normal : array-like
        Feature arrays for normal samples, shape (N_normal, D)
    X_misalign : array-like
        Feature arrays for misalignment samples, shape (N_misalign, D)
    X_unbalance : array-like
        Feature arrays for unbalance samples, shape (N_unbalance, D)
    X_looseness : array-like
        Feature arrays for looseness samples, shape (N_looseness, D)
    test_size : float, default=0.15
        Proportion of the dataset to include in the test split
    val_size : float, default=0.15
        Proportion of the dataset to include in the validation split
    random_state : int, default=42
        Random state for reproducible splits
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # === Combine Features and Labels ===

    # X_normal, X_misalign, X_unbalance, and X_looseness are feature arrays
    # Each has shape (N_i, D), where N_i is the number of samples for that fault class,
    # and D is the number of features per sample

    # Vertically stack all the feature arrays to form a single dataset X
    # Resulting shape: (N_total, D), where N_total = sum of all class sample counts
    X = np.vstack([X_normal, X_misalign, X_unbalance, X_looseness])

    # Now we create the corresponding label vector y
    # Each entry in y is an integer label indicating the class of the corresponding row in X

    # np.full(length, label) creates a label array of a specific class for all samples in that class
    y = np.concatenate([
        np.full(len(X_normal), 0),      # Label 0 for all "Normal" samples
        np.full(len(X_misalign), 1),    # Label 1 for all "Misalignment" samples
        np.full(len(X_unbalance), 2),   # Label 2 for all "Unbalance" samples
        np.full(len(X_looseness), 3)    # Label 3 for all "Looseness" samples
    ])

    # After this step:
    # - X[i] contains the feature vector of the i-th signal
    # - y[i] contains the corresponding fault type as a class label:
    #   0 = Normal, 1 = Misalignment, 2 = Unbalance, 3 = Looseness
    #
    # This labeling is necessary for supervised learning models to learn the mapping
    # from signal features to fault types (classification task)
    
    # First split: test_size% test, remaining for train+val
    X_remain, X_test, y_remain, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: from remaining, split train and val
    # Calculate val_size_adjusted to get the desired final validation proportion
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_remain, y_remain, test_size=val_size_adjusted, random_state=random_state, stratify=y_remain
    )

    # Display result
    print("✅ Dataset split complete:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
