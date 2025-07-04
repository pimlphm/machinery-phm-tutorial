from torchinfo import summary

def print_model_summary(model, input_shape=(1, 32, 12), device='cpu'):
    """
    Prints a detailed summary of the PyTorch model similar to TensorFlow's model.summary()

    Args:
        model (nn.Module): The PyTorch model instance
        input_shape (tuple): Shape of input tensor excluding batch size, e.g. (seq_len, num_features)
        device (str): Device to place the model for summary ('cpu' or 'cuda')

    Example:
        print_model_summary(my_model, input_shape=(32, 12))
    """
    model.to(device)
    summary(model, input_size=(1, *input_shape), col_names=["input_size", "output_size", "num_params", "trainable"])
