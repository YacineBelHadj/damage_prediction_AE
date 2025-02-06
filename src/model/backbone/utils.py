import torch.nn as nn
from src.model.transformation import NormLayer

# Activation function dictionary
activation_fn_dict = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
    'leakyrelu': nn.LeakyReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
}

class PrintShape(nn.Module):
    """
    Debugging layer to print the shape of tensors passing through.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name}: {x.shape}")
        return x


def build_layers(hidden_dims, activation_list=None, batch_norm=False, dropout_rate=0.0,
                          debug=False, init_fn=None, norm_layer=None,norm_layer_location=None):
    """
    Builds a customizable sequence of layers with advanced features such as custom activations and optional debugging.

    Args:
        hidden_dims (list[int]): List of layer dimensions, where each element is the number of neurons in that layer.
        activation_list (list[str], optional): List of activation functions to use per layer. Default is ['relu'] * (len(hidden_dims) - 1).
        batch_norm (bool): Whether to include BatchNorm layers after Linear layers. Default is True.
        dropout_rate (float): Dropout rate (0.0 to disable dropout). Default is 0.0.
        debug (bool): If True, adds debugging layers to print tensor shapes. Default is False.
        init_fn (callable, optional): Custom weight initialization function. Default is None.
        norm_layer (nn.Module, optional): A normalization layer to prepend (e.g., custom NormLayer). Default is None.

    Returns:
        nn.Sequential: A PyTorch Sequential container with the specified layers.
    """
    layers = []
    if not isinstance(activation_list, list):
        activation_list = [activation_list] * (len(hidden_dims) - 1)

    if norm_layer:
        assert norm_layer_location in ['pre', 'post'], "norm_layer_location must be 'pre' or 'post'"
    if norm_layer_location:
        assert norm_layer, "norm_layer must be provided if norm_layer_location is specified"
        
    if norm_layer and norm_layer_location == 'pre':
        layers.append(norm_layer)
    if not isinstance(batch_norm, list):
        batch_norm_list = [batch_norm] * (len(hidden_dims) - 1)
    assert len(batch_norm_list) == len(hidden_dims) - 1, "batch_norm_list must have len(hidden_dims) - 1"
        
    for i in range(len(hidden_dims) - 1):
        in_dim, out_dim = hidden_dims[i], hidden_dims[i + 1]
        if debug:
            layers.append(PrintShape(f"Layer {i}: {in_dim} -> {out_dim}"))
        # Main Linear Layer
        layers.append(nn.Linear(in_dim, out_dim))

        # Apply custom initialization if provided
        if init_fn:
            init_fn(layers[-1])

        # Optional BatchNorm
        if batch_norm_list[i]:
            layers.append(nn.BatchNorm1d(out_dim))

        # Activation Function
        activation_fn = activation_fn_dict.get(activation_list[i], None)
        if activation_fn:
            layers.append(activation_fn)

        # Optional Dropout
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))

        # Debugging Layer

    if norm_layer and norm_layer_location == 'post':
        layers.append(norm_layer)

    return nn.Sequential(*layers)

def build_conv_layers(
    conv_specs: list,
    activation_list=None,
    batch_norm=True,
    dropout_rate=0.0,
    debug=False,
    convtranspose=False,
    input_dim=None
):
    """
    Builds a stack of Conv1d (or ConvTranspose1d) layers with optional activation, batch norm, and dropout.

    Args:
        conv_specs (list[tuple]): List of layer specifications.
            Each tuple is (in_channels, out_channels, kernel_size, stride).
        activation_list (list[str] or str, optional): List of activation functions for each layer.
            Can be a single string (applied to all layers) or a list of strings.
        batch_norm (bool): Whether to apply batch normalization after each Conv layer.
        dropout_rate (float): Dropout rate (0.0 disables dropout).
        debug (bool): If True, adds debugging layers to print the shape of tensors passing through.
        convtranspose (bool): If True, builds ConvTranspose1d layers instead of Conv1d layers.
        input_dim (int, optional): Starting input length for shape calculations. Used for debugging.

    Returns:
        nn.Sequential: A PyTorch Sequential module containing the constructed layers.
        tuple: Final number of output channels and the computed output length (if input_dim is provided).
    """
    layers = []
    current_length = input_dim
    current_channels = conv_specs[0][0] if conv_specs else None

    # Normalize activation_list to a list
    if isinstance(activation_list, str):
        activation_list = [activation_list] * len(conv_specs)
    elif activation_list is None:
        activation_list = ['relu'] * len(conv_specs)

    for i, (in_c, out_c, k_size, stride) in enumerate(conv_specs):
        if current_channels is not None and in_c != current_channels:
            raise ValueError(
                f"Layer {i}: expected in_channels={current_channels}, but got {in_c}."
            )

        if debug:
            layers.append(PrintShape(f"Conv Layer {i}"))

        # Build either Conv1d or ConvTranspose1d
        conv_layer = nn.ConvTranspose1d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k_size,
            stride=stride,
            padding=0
        ) if convtranspose else nn.Conv1d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k_size,
            stride=stride,
            padding=0
        )
        layers.append(conv_layer)

        # Compute the new length if input_dim is provided
        if input_dim is not None:
            if convtranspose:
                current_length = (current_length - 1) * stride + k_size
            else:
                current_length = (current_length - k_size) // stride + 1
            if current_length < 1:
                raise ValueError(f"Layer {i}: resulting length went negative (check k_size, stride).")

        # Optionally add BatchNorm
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_c))

        # Add the activation function
        activation_fn = activation_fn_dict.get(activation_list[i].lower(), None)
        if activation_fn:
            layers.append(activation_fn)

        # Add dropout if specified
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))

        # Update for next layer
        current_channels = out_c

    final_channels = current_channels if conv_specs else None
    final_length = current_length if input_dim is not None else None

    return nn.Sequential(*layers), (final_channels, final_length)