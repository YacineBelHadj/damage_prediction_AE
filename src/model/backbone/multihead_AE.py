import torch 
from torch import nn
from src.model.transformation import NormLayer, UnsqueezeLayer, SqueezeLayer, StandardScaler

class GeLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))
class Swish(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return x * torch.sigmoid(x)

class AutoEncoder(nn.Module):
    def __init__(self,input_dim:int = 263 ,hidden_dim: list[int] = [128,64,32,16,8],activation_fn:str='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.hidden_dim = [input_dim] + hidden_dim
        activation_fn_dict = {'relu': nn.ReLU(), 'gelu': GeLU(), 'swish': Swish()}
        if activation_fn not in activation_fn_dict:
            raise ValueError(f"Activation function {activation_fn} not supported. Choose from {list(activation_fn_dict.keys())}")  
        self.activation_fn = activation_fn_dict[activation_fn]

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        
    def _build_encoder(self):
        encoder = nn.ModuleList()
        encoder.append(NormLayer(-22.88, -2.31))
        for i in range(len(self.hidden_dim)-1):
            encoder.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]))
            encoder.append(nn.BatchNorm1d(self.hidden_dim[i+1]))
            encoder.append(self.activation_fn) 
        return nn.Sequential(*encoder)

    def _build_decoder(self):
        decoder = nn.ModuleList()
        for i in range(len(self.hidden_dim)-1,0,-1):
            decoder.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i-1]))
            decoder.append(nn.BatchNorm1d(self.hidden_dim[i-1]))
            if i  < len(self.hidden_dim)-1:
                continue   
            decoder.append(self.activation_fn) 
        decoder.append(NormLayer(-22.88, -2.31, denormalize=True))
        return nn.Sequential(*decoder)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class PrintShape(nn.Module):    
    def __init__(self, name):
        super(PrintShape, self).__init__()
        self.name = name

    def forward(self, x):
        print(self.name, x.shape)
        return x
    
class AutoEncoderConv(nn.Module):
    def __init__(self, input_dim: int = 263,
                 conv_params: list[tuple[int, int, int, int]] = [(1, 16, 15, 3), (16, 32, 15, 5), (32, 64, 30, 1)],
                 latent_dim: int = 8,
                 activation_fn: str = 'relu'):
        super().__init__()
        self.input_dim = input_dim
        self.conv_params = conv_params
        self.latent_dim = latent_dim

        # Define activation functions
        activation_fn_dict = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'swish': nn.SiLU()}
        if activation_fn not in activation_fn_dict:
            raise ValueError(f"Activation function {activation_fn} not supported. Choose from {list(activation_fn_dict.keys())}")
        self.activation_fn = activation_fn_dict[activation_fn]
        
        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _build_encoder(self):
        layers = []
        current_dim = self.input_dim  # Initial temporal/spatial dimension
        layers.append(NormLayer(-22.88, -2.31))
        for idx, (in_channels, out_channels, kernel_size, stride) in enumerate(self.conv_params):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride))
            layers.append(self.activation_fn)
            layers.append(nn.BatchNorm1d(out_channels))  # Optional
            # Update temporal/spatial dimension
            current_dim = (current_dim - kernel_size) // stride + 1
        
        # Add a convolutional layer to map to latent_dim channels
        layers.append(nn.Conv1d(self.conv_params[-1][1], self.latent_dim, kernel_size=current_dim))
        self.pre_final_dim = current_dim  # After latent conv, temporal dimension is 1
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        reversed_conv_params = self.conv_params[::-1]
        layers = []
        # Start with mapping from latent_dim to the last out_channels in encoder
        layers.append(nn.ConvTranspose1d(self.latent_dim, reversed_conv_params[0][1], kernel_size=self.pre_final_dim))
        
        # Initialize current_dim to 1 (since latent conv output temporal dim is 1)
        current_dim = self.pre_final_dim
        
        # Iterate over conv_params in reverse to build the decoder
        for idx, (in_channels, out_channels, kernel_size, stride) in enumerate(reversed_conv_params):

            current_dim = (current_dim - 1) * stride + kernel_size
            
            layers.append(nn.ConvTranspose1d(out_channels, in_channels, kernel_size, stride))
            layers.append(self.activation_fn)
            layers.append(nn.BatchNorm1d(in_channels))  # Optional
        
        # Finally, map back to the input channels (e.g., 1)
        #  params of the final layer are the same as the first layer of the encoder 
        final_kernel_size = self.input_dim - (current_dim - 1)    

        layers.append(nn.ConvTranspose1d(self.conv_params[0][0], self.conv_params[0][0], kernel_size=final_kernel_size, stride=1))

        layers.append(NormLayer(-22.88,-2.31,denormalize=True))  # Assuming input was normalized between 0 and 1
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



         
class SimpleMLP(nn.Module):
    def __init__(self, input_dim:int, hidden_dim: list[int],min_val:float=107433,max_val:float=6568131):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_val = min_val
        self.max_val = max_val
        self.hidden_dim = [input_dim] + hidden_dim
        self.model = self._build_model()
        
    def _build_model(self):
        model = nn.ModuleList()
        for i in range(len(self.hidden_dim)-1):
            model.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]))
            model.append(nn.BatchNorm1d(self.hidden_dim[i+1]))  
            model.append(nn.LeakyReLU()) 
        model.append(nn.Linear(self.hidden_dim[-1], 1))
        model.append(NormLayer(self.max_val, self.min_val, denormalize=True))
        return nn.Sequential(*model)    
    def forward(self, x):
        return self.model(x)
        

def build_layers(input_dim, hidden_dims, activation_fn, use_norm=True, norm_layer=None, include_last_activation=True):
    """
    Utility function to build sequential layers with Linear, BatchNorm, and Activation.
    
    Args:
        input_dim (int): Input dimension.
        hidden_dims (list[int]): List of hidden layer dimensions.
        activation_fn (nn.Module): Activation function instance.
        use_norm (bool): Whether to use BatchNorm.
        norm_layer (nn.Module): Optional normalization layer to add as the first layer.
        include_last_activation (bool): Whether to include activation after the last Linear layer.
        
    Returns:
        nn.Sequential: Sequential model with the specified layers.
    """
    layers = [norm_layer] if norm_layer else []
    hidden_dims = [input_dim] + hidden_dims  # Include input_dim as the first layer
    
    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Add BatchNorm and activation except for the last layer if specified
        if i < len(hidden_dims) - 2 or include_last_activation:
            if use_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(activation_fn)
    
    return nn.Sequential(*layers)


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


