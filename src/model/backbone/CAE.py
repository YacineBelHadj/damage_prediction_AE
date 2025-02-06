import torch
import torch.nn as nn
from src.model.backbone.utils import build_conv_layers
from src.model.transformation import MultiChannelNormalization, DictStack, CutDict

class ConvAutoEncoder(nn.Module):
    """
    A 1D Convolutional AutoEncoder that uses build_conv_layers for both:
      - An encoder (normal conv)
      - A decoder (transpose conv)
    """
    def __init__(
        self,
        input_spec={'Welch_X': 264, 'Welch_Y': 264, 'Welch_Z': 264},
        encoder_specs=[(3,16,11,2), (16,32,5,2), (32,64,5,2), (64,64,5,2)],
        decoder_specs=[(64,64,11,2), (64,32,5,2), (32,16,5,2),(16,3,5,2)],
        latent_dim=32,
        encoder_activation='relu',
        decoder_activation='relu',
        batch_norm=True,
        dropout_rate=0.0,
        debug=False,
        # For example, supply min/max for each sensor if you're using 'minmax' scaling
        normalization_values = {
            'Welch_X': {'max': 7, 'min': -22},
            'Welch_Y': {'max': 7, 'min': -18},
            'Welch_Z': {'max': 8, 'min': -16},
        },
    ):
        super().__init__()

        self.latent_dim = latent_dim
        sensor_list = list(input_spec.keys())
        self.input_spec = input_spec

        # 1) Normalization + stacking
        self.normalization_layer_in = MultiChannelNormalization(
            sensor_list=sensor_list,
            normalization_type='minmax',  # e.g. 'minmax' or 'meanstd'
            statistics=normalization_values,
            denormalize=False
        )
        self.in_stack = DictStack(keys=sensor_list, dim=1, unstack=False)


        self.encoder, (final_channels, final_length) = build_conv_layers(
            conv_specs=encoder_specs,
            activation_list=encoder_activation,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            debug=debug,
            convtranspose=False,
            input_dim=264  # Starting length
        )

        # 5) Combine them into two big pipelines for convenience
        self.incomplet_encoder = nn.Sequential(
            self.normalization_layer_in,
            self.in_stack,
            self.encoder,
        )
        shape_after_encoder = self._find_output_length_of_encoder()
        self.final_layer = nn.Conv1d(final_channels, latent_dim, kernel_size=shape_after_encoder[2])
        self.full_encoder = nn.Sequential(
            self.incomplet_encoder,
            self.final_layer
        )
        self.first_deconv = nn.ConvTranspose1d(latent_dim, final_channels, kernel_size=shape_after_encoder[2])


        # Now build the actual transposed conv stack:
        self.decoder, _ = build_conv_layers(
            conv_specs=decoder_specs,
            activation_list=decoder_activation,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            debug=debug,
            convtranspose=True,
            input_dim=final_length  # The length after the encoder
        )

        # 4) Output unstack + denormalization
        self.out_stack = DictStack(keys=sensor_list, dim=1, unstack=True)
        self.normalization_layer_out = MultiChannelNormalization(
            sensor_list=sensor_list,
            normalization_type='minmax',
            statistics=normalization_values,
            denormalize=True
        )

        self.cutdict = CutDict(input_spec=input_spec)
        self.full_decoder = nn.Sequential(
            self.first_deconv,
            self.decoder,
            self.out_stack,
            self.normalization_layer_out,
            self.cutdict
        )

    def _find_output_length_of_encoder(self):
        x_in_test = {k: torch.randn(1, v) for k, v in self.input_spec.items()}
        with torch.no_grad():
            out = self.incomplet_encoder(x_in_test)
        return out.shape

    def encode(self, x_dict):
        """
        x_dict: a dict of sensor data e.g. {'Welch_X': Tensor, 'Welch_Y': Tensor, ...}
        Returns the latent embedding as a Tensor of shape (B, latent_dim, length_after_enc).
        """
        return self.full_encoder(x_dict)

    def decode(self, z):
        """
        z: a Tensor of shape (B, latent_dim, some_length).
        Returns a dict with the same keys as input_spec.
        """
        return self.full_decoder(z)

    def forward(self, x_dict):
        """
        Full forward pass:
          1) encoder => z
          2) decoder => reconstruction
          3) store into x_dict for convenience
        """
        z = self.encode(x_dict)
        out = self.decode(z)

        # Optionally store in original dict
        x_dict['reconstruction'] = out
        x_dict['latent_layer']   = z
        return x_dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TrainingConvAE(pl.LightningModule):
    """
    PyTorch Lightning module for training the convolutional autoencoder.
    """

    def __init__(
        self,
        input_spec: dict = {'Welch_X': 264, 'Welch_Y': 264, 'Welch_Z': 264},
        encoder_specs=[(3, 16, 11, 2), (16, 32, 5, 2), (32, 64, 5, 2)],
        decoder_specs=[(64, 32, 10, 2), (32, 16, 5, 2), (16, 3, 3, 2)],
        latent_dim: int = 32,
        encoder_activation: str = 'relu',
        decoder_activation: str = 'relu',
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        debug: bool = False,
        normalization_values: dict = None,
        lr: float = 1e-3,
    ):
        """
        Args:
            input_spec (dict): Dictionary mapping sensor names to their 1D input lengths.
            encoder_specs (list[tuple]): Conv1d specs for the encoder, e.g. (in_ch, out_ch, kernel, stride).
            decoder_specs (list[tuple]): ConvTranspose1d specs for the decoder, e.g. (in_ch, out_ch, kernel, stride).
            latent_dim (int): Number of latent channels in the bottleneck.
            encoder_activation (str): Activation for encoder layers (e.g., 'relu').
            decoder_activation (str): Activation for decoder layers (e.g., 'relu').
            batch_norm (bool): Whether to use batch normalization in conv blocks.
            dropout_rate (float): Dropout rate in conv blocks.
            debug (bool): If True, prints shape at each layer.
            normalization_values (dict): Min/max (or mean/std) used for normalization of each sensor.
            lr (float): Learning rate for optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_spec = input_spec
        self.lr = lr
        self.loss_fn = nn.MSELoss()

        # Instantiate the convolutional autoencoder
        self.model = ConvAutoEncoder(
            input_spec=input_spec,
            encoder_specs=encoder_specs,
            decoder_specs=decoder_specs,
            latent_dim=latent_dim,
            encoder_activation=encoder_activation,
            decoder_activation=decoder_activation,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            debug=debug,
            normalization_values=normalization_values,
        )


    def forward(self, x_dict):
        """
        Forward pass through the model.

        Args:
            x_dict (dict): Dictionary containing input tensors for each sensor.
        Returns:
            dict: Dictionary containing 'reconstruction' and 'latent_layer'.
        """
        return self.model(x_dict)

    def _compute_loss(self, x_dict, output_dict):
        """
        Computes the reconstruction loss (MSE across all sensors).

        Args:
            x_dict (dict): Original inputs {sensor_name: Tensor}.
            output_dict (dict): Model output containing 'reconstruction' dict.

        Returns:
            Tensor: Summed MSE loss over all sensors.
        """
        recon_loss = 0.0
        recon_dict = output_dict['reconstruction']
        for sensor_name in self.input_spec.keys():
            # MSELoss on each sensor channel
            recon_loss += self.loss_fn(recon_dict[sensor_name], x_dict[sensor_name])
        return recon_loss

    def training_step(self, batch, batch_idx):
        """
        Single training step.

        Args:
            batch (dict): Dict of sensor data {sensor_name: Tensor}.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Training loss.
        """
        output = self.forward(batch)
        loss = self._compute_loss(batch, output)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step.

        Args:
            batch (dict): Dict of sensor data {sensor_name: Tensor}.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Validation loss.
        """
        output = self.forward(batch)
        loss = self._compute_loss(batch, output)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and LR scheduler.

        Returns:
            dict: Dictionary with optimizer and LR scheduler (ReduceLROnPlateau).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }


if __name__ == '__main__':
    # Example usage
    B = 8
    x_dict = {
        'Welch_X': torch.randn(B, 264),
        'Welch_Y': torch.randn(B, 264),
        'Welch_Z': torch.randn(B, 264),
    }

    # Create the Lightning module
    pl_module = TrainingConvAE()

    # Forward pass (for debugging)
    out = pl_module(x_dict)
    print("Available keys in output:", out.keys())
    for k, v in out['reconstruction'].items():
        print(f"{k} => {v.shape}")
    print("Latent shape:", out['latent_layer'].shape)
