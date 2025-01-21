import torch.nn as nn
from src.model.transformation import NormLayer
import torch

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

def initialize_weights(layer):
    """
    Custom weight initialization function for linear layers.
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def build_layers(hidden_dims, activation_list=None, batch_norm=True, dropout_rate=0.0,
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
        if batch_norm:
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

class DictStack(nn.Module):
    """
    Module to either stack a dictionary of tensors into one tensor
    or unstack a single tensor back into a dictionary, depending on `unstack`.

    If unstack=False (the default):
      forward expects a dict {key: Tensor} and returns a single Tensor stacked along `dim`.

    If unstack=True:
      forward expects a single Tensor and returns a dict {key: Tensor} by splitting/unbinding along `dim`.

    Args:
        keys (list[str]): The list of keys in the dictionary, and
                          also the order in which to stack or unstack.
        dim (int): The dimension along which to stack or unstack. Default=1.
        unstack (bool): Whether to unstack (True) or stack (False). Default=False.
    """
    def __init__(self, keys, dim=1, unstack=False):
        super().__init__()
        self.keys = keys
        self.dim = dim
        self.unstack = unstack

    def forward_stack(self, x_dict: dict) -> torch.Tensor:
        """
        Takes a dict of Tensors {key: (shape...)} and stacks them along self.dim.
        For example, if each x_dict[key] is (B, L), the result might be (B, C, L)
        where C = len(self.keys) if dim=1.
        """
        # Gather the tensors in the order of self.keys
        tensors = [x_dict[k] for k in self.keys]
        # Stack along self.dim
        return torch.stack(tensors, dim=self.dim)

    def forward_unstack(self, x_tensor: torch.Tensor) -> dict:
        """
        Takes a single tensor, e.g. (B, C, L), and unbinds/unstacks it along self.dim
        into a dict {key: (B, L)} for each channel.
        """
        # Unbind along self.dim => a tuple of Tensors
        # e.g. if x_tensor.shape is (B, C, L) and dim=1, we get C slices of shape (B, L)
        splitted = torch.unbind(x_tensor, dim=self.dim)

        # Zip each slice with a corresponding key
        out_dict = {}
        for k, t in zip(self.keys, splitted):
            out_dict[k] = t
        return out_dict

    def forward(self, x):
        """
        If unstack=False, expects a dict and returns a stacked Tensor.
        If unstack=True, expects a Tensor and returns a dict.
        """
        if self.unstack:
            # x is a single tensor => unbind/unstack to dict
            return self.forward_unstack(x)
        else:
            # x is a dict => stack to a single tensor
            return self.forward_stack(x)


import torch
import torch.nn as nn

# Debugging layer to print shapes
class PrintShape(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name}: {x.shape}")
        return x

# Example activation functions
activation_fn_dict = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
    'leakyrelu': nn.LeakyReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
}

class DictStack(nn.Module):
    """
    Module to either stack a dictionary of tensors into one tensor
    or unstack a single tensor back into a dictionary, depending on `unstack`.

    If unstack=False (the default):
      forward expects a dict {key: Tensor} and returns a single Tensor stacked along `dim`.

    If unstack=True:
      forward expects a single Tensor and returns a dict {key: Tensor} by splitting/unbinding along `dim`.
    """
    def __init__(self, keys, dim=1, unstack=False):
        super().__init__()
        self.keys = keys
        self.dim = dim
        self.unstack = unstack

    def forward_stack(self, x_dict: dict) -> torch.Tensor:
        tensors = [x_dict[k] for k in self.keys]
        return torch.stack(tensors, dim=self.dim)

    def forward_unstack(self, x_tensor: torch.Tensor) -> dict:
        splitted = torch.unbind(x_tensor, dim=self.dim)
        out_dict = {}
        for k, t in zip(self.keys, splitted):
            out_dict[k] = t
        return out_dict

    def forward(self, x):
        if self.unstack:
            return self.forward_unstack(x)  # single tensor -> dict
        else:
            return self.forward_stack(x)    # dict -> single tensor


def build_conv_layers(
    conv_specs: list = [(3, 64, 5, 1), (64, 128, 3, 1)],
    activation_list: list = None,
    batch_norm: bool = True,
    dropout_rate: float = 0.0,
    debug: bool = False,
    concatenation_module: nn.Module = None,
    keys_dict = ['Welch_X', 'Welch_Y', 'Welch_Z'],
    stack_dim = 1,
    input_dim = 264,
    convtranspose = False,
):
    layers = []

    current_length = input_dim
    current_channels = conv_specs[0][0] if conv_specs else None
    if concatenation_module is not None:
        layers.append(concatenation_module)



    num_layers = len(conv_specs)
    if activation_list is None:
        activation_list = ['relu'] * num_layers
    elif isinstance(activation_list, str):
        activation_list = [activation_list] * num_layers
    else:
        assert len(activation_list) == num_layers


    for i, (in_c, out_c, k_size, stride) in enumerate(conv_specs):
        if current_channels is not None and in_c != current_channels:
            raise ValueError(
                f"Layer {i}: expected in_channels={current_channels}, got {in_c}."
            )
        if debug:
            layers.append(PrintShape(f"Conv layer {i}: (B, {in_c}, L_in)->(B, {out_c}, L_out)"))

        # Build either conv or transpose conv
        if convtranspose:
            conv = nn.ConvTranspose1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=k_size,
                stride=stride,
                padding=0
            )
            # shape formula for transpose conv (no padding/dilation/output_padding):
            # L_out = (L_in - 1)*stride + k_size
            new_length = (current_length - 1) * stride + k_size

        else:
            conv = nn.Conv1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=k_size,
                stride=stride,
                padding=0
            )
            # shape formula for standard conv (no padding/dilation):
            # L_out = floor((L_in - k_size)/stride + 1)
            new_length = (current_length - k_size) // stride + 1
            if new_length < 1:
                raise ValueError(f"Layer {i}: shape went negative (check k_size, stride).")

        layers.append(conv)
        current_length = new_length
        current_channels = out_c

        if batch_norm:
            layers.append(nn.BatchNorm1d(out_c))

        act = activation_list[i].lower()
        if act not in activation_fn_dict:
            raise ValueError(f"Unknown activation '{act}'.")
        layers.append(activation_fn_dict[act])

        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))



    final_channels = current_channels if num_layers > 0 else None
    final_length = current_length if num_layers > 0 else input_dim

    return nn.Sequential(*layers), (final_channels, final_length)
