from sklearn.preprocessing import StandardScaler, LabelEncoder
# StandardScaler: scales features to zero mean and unit variance
# LabelEncoder: converts string or integer labels into sequential integer codes
from tensorflow.keras.utils import to_categorical
# Converts integer class labels to one-hot encoded format for categorical classification


def training_data_preparation(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Preprocesses features and labels for machine learning.
    
    Args:
        X_train, X_val, X_test: Feature datasets
        y_train, y_val, y_test: Label datasets
    
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train_cat, y_val_cat, y_test_cat)
    """
    # (1)Initialize a label encoder to convert string/integer labels into 0,1,2,... classes
    le = LabelEncoder()
    # (2) Fit the encoder on training labels and transform all label sets
    # This ensures consistent mapping across train, val, and test sets
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    # (3) One-hot encode the integer labels for multi-class classification (needed by softmax)
    # Converts e.g. [0,1,2] → [[1,0,0], [0,1,0], [0,0,1]]
    y_train_cat = to_categorical(y_train_enc)
    y_val_cat = to_categorical(y_val_enc)
    y_test_cat = to_categorical(y_test_enc)
    #(1) StandardScaler transforms data to have mean 0 and variance 1 per feature
    scaler = StandardScaler()

    #(2)Fit the scaler on training data and transform all datasets using the same scale
    X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform on train
    X_val_scaled = scaler.transform(X_val)          # Transform only (no fit)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_cat, y_val_cat, y_test_cat
