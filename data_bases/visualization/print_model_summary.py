from torchinfo import summary

def print_model_summary(model, input_shape=(32, 21), device='cpu'):
    """
    Print a detailed summary of a PyTorch model (like Keras model.summary()).
    
    Args:
        model (nn.Module): PyTorch model instance
        input_shape (tuple): (seq_len, num_features) of the input
        device (str): 'cpu' or 'cuda'
    """
    try:
        model.to(device)
        print("=" * 60)
        print(f" Model Summary: {model.__class__.__name__}")
        print(f" Input shape (no batch): {input_shape}")
        print("=" * 60)

        summary(model,
                input_size=(1, *input_shape),  # batch_size=1
                col_names=["input_size", "output_size", "num_params", "trainable"],
                depth=4,
                device=device)

    except RuntimeError as e:
        print("❌ Model summary failed.")
        print("Reason:", e)
        print("💡 Hint: Check that input_shape[-1] matches model's expected input_size.")
