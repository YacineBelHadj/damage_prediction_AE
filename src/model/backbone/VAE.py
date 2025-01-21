import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.backbone.utils import build_layers
from src.model.transformation import NormLayer, UnsqueezeLayer, SqueezeLayer, StandardScaler
import pytorch_lightning as pl
# import lr_reduce on plateau
from torch.optim.lr_scheduler import ReduceLROnPlateau

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_spec: dict = {'Welch_X': 264},
                 latent_dim: int = 8,
                 hidden_dims: list[int] = [128, 64, 32],
                 activation_str: str = 'relu',
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 normalization_values: dict = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation_fn = activation_str
        self.name_input = list(input_spec.keys())[0]
        self.input_dim = input_spec[self.name_input]

        # Encoder
        self.encoder = build_layers(hidden_dims=[self.input_dim] + hidden_dims,
                                    activation_list=activation_str,
                                    batch_norm=batch_norm,
                                    dropout_rate=dropout,
                                    norm_layer=StandardScaler(mean=normalization_values[self.name_input]['mean'],
                                                         std=normalization_values[self.name_input]['std']),
                                    norm_layer_location='pre')
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var_layer = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        self.decoder = build_layers(hidden_dims=[latent_dim] + hidden_dims[::-1] + [self.input_dim],
                                     activation_list=activation_str,
                                     batch_norm=batch_norm,
                                     dropout_rate=dropout,
                                    norm_layer=StandardScaler(mean=normalization_values[self.name_input]['mean'],
                                                         std=normalization_values[self.name_input]['std'],
                                                         denormalize=True),
                                    norm_layer_location='post'
                                    )

        # Optional normalization layer

    def encode(self, x):
        x_selected = x[self.name_input]
        encoded = self.encoder(x_selected)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
   
   
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output = {self.name_input: {'reconstruction':self.decode(z), 'mu': mu, 'log_var': log_var}}
        return output

import torch

class TrainingVariationalAE(pl.LightningModule):
    def __init__(self, input_spec: dict = {'Welch_X': 264},
                 latent_dim: int = 8,
                 hidden_dims: list[int] = [128, 64, 32],
                 activation_str: str = 'relu',
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 normalization_values: dict = None,
                lr: float = 1e-3,
                kl_weight=1.0):
        super().__init__()
        self.model = VariationalAutoEncoder(input_spec = input_spec, latent_dim=latent_dim,
                                            hidden_dims=hidden_dims, activation_str=activation_str,
                                            batch_norm=batch_norm, dropout=dropout,
                                            normalization_values=normalization_values)
        self.input_spec = input_spec
        self.input_name = list(input_spec.keys())[0]
        self.input_dim = input_spec[self.input_name]
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.kl_weight = kl_weight
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        output = self(batch)[self.input_name]
        loss = self._compute_loss(x=batch[self.input_name],reconstruction=output['reconstruction'] ,
                                  mu=output['mu'], log_var=output['log_var'])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def _compute_loss(self, x, reconstruction, mu, log_var):
        # Reconstruction loss
        recon_loss = self.loss_fn(reconstruction, x)
        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return recon_loss + self.kl_weight*kl_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
class MultiHeadVariationalAutoEncoder(nn.Module):
    def __init__(self, input_spec: dict = {'Welch_X': 264,
                                         'Welch_Y': 264,
                                         'Welch_Z': 264},
                 latent_dim: int = 8,
                 separated_layer: list[int] = [128, 64, 32],
                 shared_layer: list[int] = [128, 64, 32],
                 activation_str: str = 'relu',
                 batch_norm: bool = True,
                 dropout_rate: float = 0.0,
                 normalization_values: dict = None):
        super().__init__()
        self.input_specs = input_spec
        self.latent_dim = latent_dim
        self.normalization_values = normalization_values
        self.activation_str = activation_str
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.turbine_embedding = nn.Embedding(44, 1)

        # Separate encoders for each input
        self.encoder_separated = nn.ModuleDict({
            name: build_layers(
                hidden_dims=[dim] + separated_layer,
                activation_list=activation_str,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                norm_layer=StandardScaler(
                    mean=normalization_values[name]['mean'],
                    std=normalization_values[name]['std']
                ),
                norm_layer_location='pre',
                debug=False
            )
            for name, dim in input_spec.items()
        })
        
        # Shared encoder layers after concatenation
        shared_input_dim = len(input_spec) * separated_layer[-1]
        self.encoder_shared = build_layers(
            hidden_dims=[shared_input_dim] + shared_layer,
            activation_list=activation_str,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            norm_layer=None,  # Assuming normalization is handled in separate encoders
        )

        # Latent variables
        self.mu_layer = nn.Linear(shared_layer[-1], latent_dim)
        self.log_var_layer = nn.Linear(shared_layer[-1], latent_dim)
        
        # Shared decoder layers before separation
        shared_layer_decoder = [latent_dim]+shared_layer[::-1]
        separated_layer_decoder = [shared_layer_decoder[-1]+1] + separated_layer[::-1]
        
        self.decoder_shared = build_layers(
            hidden_dims= shared_layer_decoder,
            activation_list=activation_str,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            norm_layer=None,         
        )
        
        # Separate decoders for each output
        self.decoder_separated = nn.ModuleDict({
            name: build_layers(
                hidden_dims=separated_layer_decoder + [dim],  # Start with 128
                activation_list=['relu'] * (len(separated_layer)) + [None],  # No activation for last layer
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                norm_layer=StandardScaler(
                    mean=normalization_values[name]['mean'],
                    std=normalization_values[name]['std'],
                    denormalize=True
                ),
                debug=False,
                norm_layer_location='post',
            )
            for name, dim in input_spec.items()
        })
        
    def encode(self, x: dict):
        """
        Encodes each input separately, concatenates them, and passes through shared layers to obtain mu and log_var.
        """
        # Encode each input separately
        encoded_separated = []
        for name, encoder in self.encoder_separated.items():
            encoded = encoder(x[name])
            encoded_separated.append(encoded)
        
        # Concatenate all encoded features
        concatenated = torch.cat(encoded_separated, dim=1)
        
        # Pass through shared encoder layers
        shared_encoded = self.encoder_shared(concatenated)
        
        # Obtain mu and log_var for latent space
        mu = self.mu_layer(shared_encoded)
        log_var = self.log_var_layer(shared_encoded)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z,turbine_id):
        """
        Decodes the latent vector z into each output modality.
        """
        # Pass through shared decoder layers
        turb_emb = self.turbine_embedding(turbine_id)
        shared_decoded = self.decoder_shared(z)
        shared_decoded_cond = torch.cat([shared_decoded,turb_emb],dim=1)        
        # Decode each modality separately
        decoded = {}
        for name, decoder in self.decoder_separated.items():
            decoded[name] = decoder(shared_decoded_cond)
        
        return decoded
    
    def forward(self, x: dict):
        """
        Forward pass through the VAE.
        Returns a dictionary containing reconstructions, mu, and log_var.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z,x['turbine_name'])
        
        # Construct output dictionary
        output = {
            'reconstruction': reconstructed,
            'mu': mu,
            'log_var': log_var
        }
        
        return output


class TrainingMultiHeadVAE(pl.LightningModule):
    def __init__(self, input_spec: dict = {'Welch_X': 264,
                                         'Welch_Y': 264,
                                         'Welch_Z': 264},
                 latent_dim: int = 8,
                 separated_layer: list[int] = [128, 64, 32],
                 shared_layer: list[int] = [128, 64, 32],
                 activation_str: str = 'relu',
                 batch_norm: bool = True,
                 dropout_rate: float = 0.0,
                 normalization_values: dict = None,
                 lr: float = 1e-3,
                 kl_weight: float = 1.0):
        """
        PyTorch Lightning module for training the MultiHead VAE.

        Args:
            input_spec (dict): Dictionary with input names as keys and their dimensions as values.
            latent_dim (int): Dimension of the latent space.
            separated_layer (list[int]): Hidden layer dimensions for separated encoders/decoders.
            shared_layer (list[int]): Hidden layer dimensions for shared encoders/decoders.
            activation_str (str): Activation function to use.
            batch_norm (bool): Whether to use batch normalization.
            dropout_rate (float): Dropout rate.
            normalization_values (dict): Normalization parameters for each input.
            lr (float): Learning rate.
            kl_weight (float): Weight for the KL divergence loss.
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_spec = input_spec
        self.model = MultiHeadVariationalAutoEncoder(
            input_spec=input_spec,
            latent_dim=latent_dim,
            separated_layer=separated_layer,
            shared_layer=shared_layer,
            activation_str=activation_str,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            normalization_values=normalization_values
        )

        self.lr = lr
        self.kl_weight = kl_weight
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (dict): Dictionary containing input tensors for each modality.

        Returns:
            dict: Dictionary containing reconstructions, mu, and log_var.
        """
        return self.model(x)

    def _compute_loss(self, x, output):
        """
        Computes the combined reconstruction and KL divergence loss.

        Args:
            x (dict): Original inputs.
            output (dict): Output from the model containing 'reconstruction', 'mu', and 'log_var'.

        Returns:
            Tensor: Combined loss.
        """
        # Reconstruction loss: sum MSE loss for each modality
        recon_loss = 0
        for name in self.input_spec.keys():
            recon_loss += self.loss_fn(output['reconstruction'][name], x[name])
        
        # KL Divergence loss
        mu = output['mu']
        log_var = output['log_var']
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x[next(iter(x))].size(0)
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Training loss.
        """
        output = self(batch)
        loss = self._compute_loss(batch, output)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Validation loss.
        """
        output = self(batch)
        loss = self._compute_loss(batch, output)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            Optimizer: Adam optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }