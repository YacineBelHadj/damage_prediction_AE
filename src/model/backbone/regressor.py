from src.model.backbone.utils import build_layers
from src.model.transformation import NormLayer, StandardScaler
import pytorch_lightning as pl
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.model.backbone.AE import TrainingMultiHeadAE

class MLPRegressor(nn.Module):
    def __init__(self, input_spec: dict = {'latent': 2}, 
                 hidden_dims: list[int] = [32, 16], 
                 activation: str = 'relu', 
                 dropout: float = 0.0, 
                 batch_norm: bool = False,
                 normalization_values: dict = None
                 ):
        """
        A simple MLP regressor using build_layers from your codebase.

        :param input_dim:   Number of input features (e.g. VAE latent dim).
        :param hidden_dims: List of hidden layer sizes. [32,16] means two layers: 32 -> 16 -> output.
        :param activation:  Activation function string accepted by build_layers, e.g. 'relu' or 'tanh'.
        :param batch_norm:  Whether or not to use batch normalization in the hidden layers.
        """
        super().__init__()
        self.input_spec = input_spec
        input_dims = sum(list(input_spec.values()))
        hidden_dims_all = [input_dims] + hidden_dims + [1]
        activation = [activation] * (len(hidden_dims)) + [None]

        self.net = build_layers(
            hidden_dims=hidden_dims_all,         # e.g. [32,16,1]
            activation_list=activation,
            batch_norm=batch_norm,
            dropout_rate=dropout,
            norm_layer=NormLayer(max_val=normalization_values['DEM']['max'],
                                 min_val=normalization_values['DEM']['min'],
                                 denormalize=True),             
            norm_layer_location='post',
            debug=False
        )

    def forward(self, x):
        
        input_tensor = torch.cat([x[key] for key in self.input_spec], dim=1)
        out = {'dem_pred': self.net(input_tensor).flatten()}
        x.update(out)
        return x
        
    



class TrainRegressor(pl.LightningModule):
    def __init__(self, input_spec: dict = {'latent': 2}, 
                 hidden_dims: list[int] = [32, 16], 
                 activation: str = 'relu', 
                 dropout: float = 0.0, 
                 batch_norm: bool = False,
                 normalization_values: dict = None,
                 lr: float = 1e-3,
                 lr_encoder: float = 1e-3,
                 output_key: str = 'DEM',
                 vae_checkpoint: str = None,
                 

    ):
        """

        """
        super().__init__()
        self.save_hyperparameters(ignore=['vae','regrossor'])
        self.output_key = output_key
        # 1) Load the pretrained VAE from checkpoint
        self.vae = TrainingMultiHeadAE.load_from_checkpoint(vae_checkpoint)

        # 2) Build the regressor MLP
        self.regressor = MLPRegressor(input_spec=input_spec,
                                        hidden_dims=hidden_dims,
                                        activation=activation,
                                        dropout=dropout,
                                        batch_norm=batch_norm,
                                        normalization_values=normalization_values)

        # 3) Other attributes
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.lr_encoder = lr_encoder


    def forward(self, x):
        """
        Forward pass:
          1) Use latent_extractor(...) to get z from the VAE
          2) Regress from z -> predicted output
        """
        out = self.vae(x)
        out = self.regressor(out)
        x.update(out)
        return out

    def _common_step(self, batch, batch_idx):
        out = self(batch)
        return self.loss_fn(out[self.output_key], out["dem_pred"])

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=self.lr)
        if self.lr_encoder:
            optimizer.add_param_group({'params': self.vae.model.encoder_separated.parameters(), 'lr': self.lr_encoder})
            optimizer.add_param_group({'params': self.vae.model.encoder_shared.parameters(), 'lr': self.lr_encoder})
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

                 