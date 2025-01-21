import torch 
from torch import nn
from src.model.transformation import NormLayer, UnsqueezeLayer, SqueezeLayer, StandardScaler
from src.model.backbone.utils import build_layers, GeLU, Swish

class MultiEncoder(nn.Module):
    """Submodule for the MultiModalAutoEncoder model."""
    def __init__(self, input_dims: dict, separated_layer: list[int], shared_layer: list[int],
                 activation_str: str = 'relu', normalization_values: dict = None): 
        super().__init__()
        self.input_dims = input_dims
        self.shared_layer_dims = shared_layer
        self.normalization_values = normalization_values

        # Activation function dictionary
        activation_fn_dict = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'leakyrelu': nn.LeakyReLU()}
        self.activation_fn = activation_fn_dict[activation_str]
        
        # Individual encoders
        self.encoders = nn.ModuleDict({
            name: build_layers(
                input_dim=dim, 
                hidden_dims=separated_layer, 
                activation_fn=self.activation_fn, 
                norm_layer=NormLayer(normalization_values[name]['max'], normalization_values[name]['min'])
            )
            for name, dim in input_dims.items()
        })

        # Shared interaction layer
        self.shared_layer_input = len(input_dims) * separated_layer[-1]
        self.shared_layer = build_layers(self.shared_layer_input, shared_layer, self.activation_fn)

    def forward(self, x: dict):
        """Forward pass."""
        # Iterate over encoders and use the corresponding inputs
        encoded_features = [encoder(x[name]) for name, encoder in self.encoders.items()]
        
        # Concatenate encoded features and pass them through the shared layer
        shared_input = torch.cat(encoded_features, dim=1)
        return self.shared_layer(shared_input)


class MultiDecoder(nn.Module):
    """Symmetric MultiDecoder to reconstruct inputs from shared representation with denormalization."""
    def __init__(self, output_dims: dict, separated_layer: list[int], shared_layer: list[int],
                 normalization_values: dict, activation_str: str = 'relu'):
        super().__init__()
        self.output_dims = output_dims
        self.separated_layer = separated_layer
        
        # Activation function dictionary
        activation_fn_dict = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'leakyrelu': nn.LeakyReLU()}
        self.activation_fn = activation_fn_dict[activation_str]
        
        # Shared reverse interaction layer
        shared_output_dim = len(output_dims) * separated_layer[-1]
        self.reverse_shared_layer = build_layers(shared_layer[-1], shared_layer[::-1] + [shared_output_dim], 
                                                 self.activation_fn, include_last_activation=False)
        
        # Individual decoders
        self.decoders = nn.ModuleDict({
            name: nn.Sequential(
                build_layers(separated_layer[-1], separated_layer[::-1] + [dim], self.activation_fn, include_last_activation=False),
                NormLayer(normalization_values[name]['max'], normalization_values[name]['min'], denormalize=True)
            )
            for name, dim in output_dims.items()
        })

    def forward(self, shared_output):
        """Forward pass."""
        reversed_features = self.reverse_shared_layer(shared_output)
        batch_size = shared_output.size(0)
        split_features = torch.split(reversed_features, self.separated_layer[-1], dim=1)
        
        decoded_outputs = {
            name: self.decoders[name](split_features[i])
            for i, name in enumerate(self.output_dims.keys())
        }
        return decoded_outputs


class MultiModalAutoEncoder(nn.Module):
    """MultiModal AutoEncoder that combines MultiEncoder and MultiDecoder for reconstruction."""
    def __init__(self, input_dims: dict, separated_layer: list[int], shared_layer: list[int], 
                 normalization_values: dict, activation_str: str = 'relu'):
        super().__init__()
        self.input_dims = input_dims
        self.separated_layer = separated_layer
        self.shared_layer = shared_layer
        self.normalization_values = normalization_values
        self.activation_str = activation_str
        self.encoder = MultiEncoder(
            input_dims=input_dims, 
            separated_layer=separated_layer, 
            shared_layer=shared_layer, 
            activation_str=activation_str, 
            normalization_values=normalization_values
        )

        self.decoder = MultiDecoder(
            output_dims=input_dims, 
            separated_layer=separated_layer, 
            shared_layer=shared_layer, 
            normalization_values=normalization_values, 
            activation_str=activation_str
        )
    
    def forward(self, x: dict, return_latent=False):
        latent_representation = self.encoder(x)
        reconstructed_outputs = self.decoder(latent_representation)
        if return_latent:
            return reconstructed_outputs, latent_representation
        return reconstructed_outputs


import torch
from torch import nn




class MultiModalRegressor(nn.Module):
    """
    MultiModal Regressor where inputs specified in `not_encoder` directly connect to the shared layer.
    """
    def __init__(self, input_dims: dict, separated_layer: list[int], shared_layer: list[int],
                 normalization_values: dict = None, activation_str: str = 'relu', not_encoder: list = [],
                 output_name:str='DEM'):
        """
        Args:
            input_dims (dict): Input dimensions per channel.
            separated_layer (list[int]): Layer sizes for individual encoders.
            shared_layer (list[int]): Layer sizes for the shared layer.
            normalization_values (dict): Normalization values for input channels.
            activation_str (str): Activation function.
            not_encoder (list): List of channels to bypass the encoder.
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.not_encoder = set(not_encoder)  # Convert to set for faster lookups
        self.normalization_values = normalization_values
        self.output_name = output_name

        # Activation function
        activation_fn_dict = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'leakyrelu': nn.LeakyReLU()}
        self.activation_fn = activation_fn_dict[activation_str]

        # Encoders for channels not in the 'not_encoder' list
        self.encoders = nn.ModuleDict({
            name: build_layers(
                input_dim=dim, 
                hidden_dims=separated_layer, 
                activation_fn=self.activation_fn,
                norm_layer=StandardScaler(mean=normalization_values[name]['mean'], std = normalization_values[name]['std'])
            ) if name not in self.not_encoder else nn.Identity()
            for name, dim in input_dims.items()
        })
        
        # Shared layer input calculation
        self.shared_layer_input = sum([
            separated_layer[-1] if name not in self.not_encoder else dim
            for name, dim in input_dims.items()
        ])
        self.shared_layer = build_layers(self.shared_layer_input, shared_layer, self.activation_fn, include_last_activation=False)

        # Final regression head
        self.regression_head = nn.Linear(shared_layer[-1], 1)  
        self.denorm = NormLayer(normalization_values[output_name]['max'], normalization_values[output_name]['min'], denormalize=True)

    def forward(self, x: dict):
        """
        Forward pass.
        Args:
            x (dict): Input dictionary with channel names and tensors.
        Returns:
            torch.Tensor: Regression output.
        """
        # Process inputs: bypass encoders for 'not_encoder' keys, use encoders otherwise
        encoded_features = [encoder_(x[name]) for name, encoder_ in self.encoders.items()]
                            

        # Concatenate all features
        shared_input = torch.cat(encoded_features, dim=1)

        # Shared layer and regression head
        shared_output = self.shared_layer(shared_input)
        regression_output = self.regression_head(shared_output)
        regression_output = self.denorm(regression_output)
        return regression_output


