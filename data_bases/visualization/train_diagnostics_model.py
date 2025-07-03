from tensorflow.keras.optimizers import Adam
# Adam is an efficient gradient-based optimizer used to train deep learning models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Callbacks are helper tools that automatically monitor training and take actions like saving the best model or stopping early when performance stops improving.

def train_diagnostics_model(model, X_train_scaled, y_train_cat, X_val_scaled, y_val_cat):
    """
    Compile and train the model with callbacks for saving best model and early stopping
    """
    # (1) Compile the model with Adam optimizer, categorical crossentropy loss (for softmax), and accuracy metric
    model.compile(
        optimizer=Adam(1e-3),                  # Use Adam optimizer (a smart gradient descent)
        loss='categorical_crossentropy',       # Use this for multi-class classification with softmax
        metrics=['accuracy']                   # Show accuracy (%) during training
    )

    # === Train with callback ===
    # (2)Save the best model (only when validation accuracy improves)
    checkpoint = ModelCheckpoint(
        "best_model_diagnostics.h5",           # File to save model
        monitor='val_accuracy',                # Look at validation accuracy
        save_best_only=True,                   # Only save if it's the best so far
        verbose=1                              # Print a message when model is saved
    )

    # (3)Stop training early if validation accuracy stops improving
    early_stop = EarlyStopping(
        monitor='val_accuracy',                # Also monitor validation accuracy
        patience=10,                           # Wait 10 rounds before stopping if no improvement
        restore_best_weights=True,             # Go back to best model after stopping
        verbose=1                              # Print message when early stopping is triggered
    )

    history = model.fit(
        X_train_scaled, y_train_cat,                # Training features and one-hot encoded labels
        validation_data=(X_val_scaled, y_val_cat),  # Validation set to evaluate performance after each epoch
        epochs=50,                                  # Train for up to 50 full passes through the data
        batch_size=32,                              # Use 32 samples per training step (mini-batch)
        callbacks=[checkpoint,early_stop],                     # Apply the ModelCheckpoint callback to save best model
        verbose=2                                   # Print 1 line per epoch (clean summary output)
    )
    
    return history

