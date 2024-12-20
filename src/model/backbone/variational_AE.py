import torch
import torch.nn as nn
from src.model.transformation import NormLayer

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
    layer_dims = [input_dim] + hidden_dims
    
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2 or include_last_activation:
            if use_norm:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            layers.append(activation_fn)
    
    return nn.Sequential(*layers)

class MultiEncoder(nn.Module):
    """
    Encoder for the Variational MultiModal AutoEncoder.
    Encodes multiple inputs into a shared latent representation with mean and log-variance.
    """
    def __init__(self, input_dims, separated_layer, shared_layer, latent_dim, activation_str='relu', normalization_values=None):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.normalization_values = normalization_values

        # Activation function mapping
        activation_fn_dict = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'leakyrelu': nn.LeakyReLU()}
        self.activation_fn = activation_fn_dict[activation_str]

        # Individual encoders for each input
        self.encoders = nn.ModuleDict({
            name: build_layers(
                dim, separated_layer, self.activation_fn, 
                norm_layer=NormLayer(normalization_values[name]['max'], normalization_values[name]['min'])
            )
            for name, dim in input_dims.items()
        })

        # Shared interaction layer
        shared_input_dim = len(input_dims) * separated_layer[-1]
        self.shared_layer = build_layers(shared_input_dim, shared_layer, self.activation_fn)

        # Latent layers for mean and log-variance
        self.mu_layer = nn.Linear(shared_layer[-1], latent_dim)
        self.log_var_layer = nn.Linear(shared_layer[-1], latent_dim)

    def forward(self, x):
        """
        Forward pass: Encodes inputs into mean and log-variance for latent space sampling.
        """
        encoded_features = [encoder(x[name]) for name, encoder in self.encoders.items()]
        shared_input = torch.cat(encoded_features, dim=1)
        shared_output = self.shared_layer(shared_input)
        mu = self.mu_layer(shared_output)
        log_var = self.log_var_layer(shared_output)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        """Applies the reparameterization trick to sample from the latent distribution."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

class MultiDecoder(nn.Module):
    """
    Decoder for the Variational MultiModal AutoEncoder.
    Decodes the latent representation into multiple reconstructed outputs.
    """
    def __init__(self, latent_dim, output_dims, separated_layer, shared_layer, normalization_values, activation_str='relu'):
        super().__init__()
        self.output_dims = output_dims

        # Activation function mapping
        activation_fn_dict = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'leakyrelu': nn.LeakyReLU()}
        self.activation_fn = activation_fn_dict[activation_str]

        # Shared reverse layer
        shared_output_dim = len(output_dims) * separated_layer[-1]
        self.reverse_shared_layer = build_layers(latent_dim, shared_layer[::-1] + [shared_output_dim], self.activation_fn, include_last_activation=False)

        # Individual decoders for each output
        self.decoders = nn.ModuleDict({
            name: nn.Sequential(
                build_layers(separated_layer[-1], separated_layer[::-1] + [dim], self.activation_fn, include_last_activation=False),
                NormLayer(normalization_values[name]['max'], normalization_values[name]['min'], denormalize=True)
            )
            for name, dim in output_dims.items()
        })

    def forward(self, shared_output):
        """
        Forward pass: Decodes latent representation into multiple outputs.
        """
        reversed_features = self.reverse_shared_layer(shared_output)
        split_features = torch.split(reversed_features, self.decoders[next(iter(self.decoders))][0][0].in_features, dim=1)
        return {name: decoder(split_features[i]) for i, (name, decoder) in enumerate(self.decoders.items())}

class VariationalMultiModalAutoEncoder(nn.Module):
    """
    Variational MultiModal AutoEncoder combining the encoder and decoder for end-to-end training.
    """
    def __init__(self, input_dims, separated_layer, shared_layer, latent_dim, normalization_values, activation_str='relu'):
        super().__init__()
        self.encoder = MultiEncoder(input_dims, separated_layer, shared_layer, latent_dim, activation_str, normalization_values)
        self.decoder = MultiDecoder(latent_dim, input_dims, separated_layer, shared_layer, normalization_values, activation_str)

    def forward(self, x, return_latent=False):
        """
        Forward pass: Encodes inputs, samples latent representation, and decodes back into outputs.
        """
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterize(mu, log_var)
        reconstructed_outputs = self.decoder(z)
        return (reconstructed_outputs, mu, log_var) if return_latent else (reconstructed_outputs, mu, log_var)
