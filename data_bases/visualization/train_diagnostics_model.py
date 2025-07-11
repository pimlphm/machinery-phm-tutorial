from tensorflow.keras.optimizers import Adam
# Adam is an efficient gradient-based optimizer used to train deep learning models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Callbacks are helper tools that automatically monitor training and take actions like saving the best model or stopping early when performance stops improving.

def train_diagnostics_model(model, X_train_scaled, y_train_cat, X_val_scaled, y_val_cat, model_save_path="best_model_diagnostics.h5"):
    """
    Compile and train the multi-output model with callbacks for saving best model and early stopping
    """
    # (1) Compile the multi-output model with Adam optimizer
    # The model has two outputs: reconstructed input and class probability
    model.compile(
        optimizer=Adam(1e-3),                  # Use Adam optimizer (a smart gradient descent)
        loss={
            'reconstructed_input': 'mse',      # Mean Squared Error for autoencoder reconstruction
            'class_probability': 'categorical_crossentropy'  # Categorical crossentropy for classification
        },
        loss_weights={
            'reconstructed_input': 0.5,        # Weight for reconstruction loss
            'class_probability': 1.0           # Weight for classification loss (higher priority)
        },
        metrics={
            'reconstructed_input': ['mae'],     # Mean Absolute Error for reconstruction
            'class_probability': ['accuracy']  # Accuracy for classification
        }
    )

    # === Train with callback ===
    # (2)Save the best model (only when validation classification accuracy improves)
    checkpoint = ModelCheckpoint(
        model_save_path,                       # File to save model (now as parameter)
        monitor='val_class_probability_accuracy',  # Look at validation classification accuracy
        save_best_only=True,                   # Only save if it's the best so far
        verbose=1                              # Print a message when model is saved
    )

    # (3)Stop training early if validation classification accuracy stops improving
    early_stop = EarlyStopping(
        monitor='val_class_probability_accuracy',  # Monitor validation classification accuracy
        patience=10,                           # Wait 10 rounds before stopping if no improvement
        restore_best_weights=True,             # Go back to best model after stopping
        verbose=1                              # Print message when early stopping is triggered
    )

    history = model.fit(
        X_train_scaled, 
        {
            'reconstructed_input': X_train_scaled,  # Target for reconstruction is the input itself
            'class_probability': y_train_cat        # Target for classification is one-hot encoded labels
        },
        validation_data=(
            X_val_scaled, 
            {
                'reconstructed_input': X_val_scaled,  # Validation reconstruction target
                'class_probability': y_val_cat        # Validation classification target
            }
        ),
        epochs=50,                                  # Train for up to 50 full passes through the data
        batch_size=32,                              # Use 32 samples per training step (mini-batch)
        callbacks=[checkpoint,early_stop],          # Apply the ModelCheckpoint and EarlyStopping callbacks
        verbose=2                                   # Print 1 line per epoch (clean summary output)
    )
    
    return history
