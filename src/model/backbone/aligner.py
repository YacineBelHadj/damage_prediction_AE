import torch
from torch import nn
from torch.autograd import Function
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

############################################
# 1. Gradient Reversal Function (GRL)
############################################
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)  # Identity in forward pass

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient by multiplying by -lambda_
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_: float = 1.0):
    """
    A helper function that applies the gradient reversal operation.
    """
    return GradientReversalFunction.apply(x, lambda_)

############################################
# 2. Turbine Discriminator Module
############################################
class TurbineDiscriminator(nn.Module):
    """
    A simple classifier that predicts turbine type from a latent vector.
    Note: The gradient reversal is applied externally (via `grad_reverse`).
    """
    def __init__(self, latent_dim, num_turbines):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_turbines)
        )
        
    def forward(self, z):
        return self.net(z)

############################################
# 3. DEC-Inspired Clustering Loss (Unsupervised)
############################################
class DECClusteringLoss(nn.Module):
    """
    This loss learns a set of cluster centers (prototypes) and encourages
    the latent embeddings to form clusters. It does so by computing a soft
    assignment (using a Student's t-distribution) of each embedding to the
    cluster centers and then minimizing the KL divergence between this assignment
    and a sharpened target distribution. Note that no label information is used.
    """
    def __init__(self, n_clusters, latent_dim, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, embeddings):
        """
        Args:
            embeddings: Tensor of shape (batch_size, latent_dim)
        Returns:
            A scalar clustering loss.
        """
        # Compute pairwise squared Euclidean distances between embeddings and centers.
        # Resulting shape: (batch_size, n_clusters)
        expanded_embeddings = embeddings.unsqueeze(1)            # (B, 1, D)
        expanded_centers = self.cluster_centers.unsqueeze(0)         # (1, n_clusters, D)
        distances = torch.sum((expanded_embeddings - expanded_centers)**2, dim=2)
        
        # Compute soft assignments with a Student's t-distribution.
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize over clusters

        # Compute the target distribution p.
        # (This formulation helps to emphasize confident assignments.)
        f = torch.sum(q, dim=0, keepdim=True)  # shape: (1, n_clusters)
        p = q ** 2 / (f + 1e-8)
        p = p / torch.sum(p, dim=1, keepdim=True)
        
        # Compute KL divergence loss between target distribution p and soft assignments q.
        loss = torch.mean(torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)), dim=1))
        return loss

############################################
# 4. MultiHeadAutoEncoderAndClassifier
############################################
# (Assuming you have the following helper functions available)
from src.model.backbone.utils import build_layers
from src.model.transformation import StandardScaler

class MultiHeadAutoEncoderAndClassifier(nn.Module):
    def __init__(self, 
                 input_spec: dict = {'Welch_X': 264, 'Welch_Y': 264, 'Welch_Z': 264},
                 latent_dim: int = 8,
                 separated_layer: list[int] = [128, 64, 32],
                 shared_layer: list[int] = [128, 64, 32],
                 activation_str: str = 'relu',
                 batch_norm: bool = True,
                 dropout_rate: float = 0.0,
                 normalization_values: dict = None,
                 lambda_adv: float = 1.0  # Default GRL strength
                ):
        super().__init__()
        self.input_spec = input_spec
        self.latent_dim = latent_dim
        self.normalization_values = normalization_values
        self.activation_str = activation_str
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.lambda_adv = lambda_adv

        # Separate encoders for each modality.
        self.encoder_separated = nn.ModuleDict({
            name: build_layers(
                hidden_dims=[dim] + separated_layer,
                activation_list=activation_str,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                norm_layer=StandardScaler(mean=normalization_values[name]['mean'],
                                          std=normalization_values[name]['std']),
                norm_layer_location='pre',
                debug=False
            )
            for name, dim in input_spec.items()
        })

        # Shared encoder layers.
        shared_input_dim = len(input_spec) * separated_layer[-1]
        self.encoder_shared = build_layers(
            hidden_dims=[shared_input_dim] + shared_layer,
            activation_list=activation_str,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            norm_layer=None
        )

        # Latent layer.
        self.latent_layer = nn.Linear(shared_layer[-1], latent_dim)

        # Shared decoder layers.
        shared_layer_decoder = [latent_dim] + shared_layer[::-1]
        separated_layer_decoder = [shared_layer_decoder[-1]] + separated_layer[::-1]
        self.decoder_shared = build_layers(
            hidden_dims=shared_layer_decoder,
            activation_list=activation_str,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            norm_layer=None
        )
        # Separate decoders for each modality.
        self.decoder_separated = nn.ModuleDict({
            name: build_layers(
                hidden_dims=separated_layer_decoder + [dim],
                activation_list=['relu'] * (len(separated_layer)) + [None],
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                norm_layer=StandardScaler(mean=normalization_values[name]['mean'],
                                          std=normalization_values[name]['std'],
                                          denormalize=True),
                debug=False,
                norm_layer_location='post'
            )
            for name, dim in input_spec.items()
        })

        # Turbine classifier.
        self.classifier = TurbineDiscriminator(latent_dim, num_turbines=44)
        
    def encode(self, x: dict):
        # Encode each modality separately.
        encoded_list = []
        for name, encoder in self.encoder_separated.items():
            encoded_list.append(encoder(x[name]))
        concatenated = torch.cat(encoded_list, dim=1)
        shared_encoded = self.encoder_shared(concatenated)
        latent = self.latent_layer(shared_encoded)
        return latent
    
    def decode(self, latent):
        shared_decoded = self.decoder_shared(latent)
        outputs = {name: decoder(shared_decoded) for name, decoder in self.decoder_separated.items()}
        return outputs
    
    def forward(self, x: dict, lambda_adv=None):
        """
        Forward pass:
          - Compute latent representation.
          - Reconstruct the inputs.
          - Apply the GRL (with strength lambda_adv) to the latent features before classification.
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        if lambda_adv is None:
            lambda_adv = self.lambda_adv
        # Apply GRL so that the encoder receives reversed gradients.
        turbine_pred = self.classifier(grad_reverse(latent, lambda_adv))
        return {
            'reconstruction': reconstruction,
            'latent': latent,
            'turbine_prediction': turbine_pred
        }

############################################
# 5. Lightning Module with a Single Optimizer
############################################
class TrainingMultiHeadAEAligned(pl.LightningModule):
    def __init__(self, 
                 input_spec: dict = {'Welch_X': 264, 'Welch_Y': 264, 'Welch_Z': 264},
                 latent_dim: int = 8,
                 separated_layer: list[int] = [128, 64, 32],
                 shared_layer: list[int] = [128, 64, 32],
                 activation_str: str = 'relu',
                 batch_norm: bool = True,
                 dropout_rate: float = 0.0,
                 normalization_values: dict = None,
                 lr_ae: float = 1e-3,
                 adv_loss_weight: float = 1.0,      # Weight for the adversarial (turbine) loss
                 lambda_adv: float = 1.0,           # Default GRL strength
                 clustering_loss_weight: float = 0,  # Weight for the clustering loss
                 n_clusters: int = 5                # Number of clusters for unsupervised behavior grouping
                ):
        """
        Lightning module that trains the autoencoder to reconstruct inputs while forcing the
        encoder to become turbine-invariant (via adversarial training) and to learn a clustered
        latent space that captures different wind turbine behaviors in an unsupervised manner.
        """
        super().__init__()
        # Save hyperparameters (excluding large objects)
        self.save_hyperparameters(ignore=["normalization_values"])
        self.input_spec = input_spec
        self.lr_ae = lr_ae
        self.adv_loss_weight = adv_loss_weight
        self.clustering_loss_weight = clustering_loss_weight

        self.model = MultiHeadAutoEncoderAndClassifier(
            input_spec=input_spec,
            latent_dim=latent_dim,
            separated_layer=separated_layer,
            shared_layer=shared_layer,
            activation_str=activation_str,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            normalization_values=normalization_values,
            lambda_adv=lambda_adv
        )

        self.recon_loss_fn = nn.MSELoss()
        self.class_loss_fn = nn.CrossEntropyLoss()
        # Unsupervised clustering loss (DEC-inspired)
        self.clustering_loss_fn = DECClusteringLoss(n_clusters=n_clusters, latent_dim=latent_dim)
        self.p = 0.0  # For scheduling GRL strength.
    
    def forward(self, x):   
        # Update the input dict with the model outputs.
        res = self.model(x)
        x.update(res)
        return x
        
    def compute_grl_lambda(self, batch_idx):
        """
        Compute the GRL lambda based on training progress:
            p = current_epoch / max_epochs,
            lambda = 2/(1+exp(-10*p)) - 1.
        """
        p = float(self.current_epoch) / float(self.trainer.max_epochs)
        self.p = p
        grl_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        return grl_lambda

    def _compute_recon_loss(self, batch, recon):
        loss = 0.0
        for name in self.input_spec.keys():
            loss += self.recon_loss_fn(recon[name], batch[name])
        return loss

    def training_step(self, batch, batch_idx):
        """
        Expects batch to be a dict with:
          - Modality inputs (keys in self.input_spec)
          - Turbine labels under 'turbine_name' (used only for the adversarial loss)
        """
        grl_lambda = self.compute_grl_lambda(batch_idx)
        res = self(batch)
        latent = res['latent']
        
        # --- Reconstruction Loss ---
        loss_recon = self._compute_recon_loss(batch, res['reconstruction'])
        
        # --- Adversarial Losses ---
        # Encoder branch: apply GRL to force turbine invariance.
        latent_for_ae = latent.clone()  # Clone to avoid in-place modifications.
        classifier_output_ae = self.model.classifier(grad_reverse(latent_for_ae, grl_lambda))
        loss_adv_encoder = self.class_loss_fn(classifier_output_ae, batch['turbine_name'])
        loss_ae = loss_recon + self.adv_loss_weight * loss_adv_encoder
        
        # Classifier branch: update classifier with detached latent.
        classifier_output_adv = self.model.classifier(latent.detach())
        loss_adv_classifier = self.class_loss_fn(classifier_output_adv, batch['turbine_name'])
        
        # --- Clustering Loss (Unsupervised) ---
        loss_cluster = self.clustering_loss_fn(latent)
        
        # --- Total Loss ---
        total_loss = loss_ae + loss_adv_classifier + self.clustering_loss_weight * loss_cluster

        # Logging
        self.log("train_recon_loss", loss_recon, prog_bar=True)
        self.log("train_adv_loss_encoder", loss_adv_encoder, prog_bar=True)
        self.log("train_loss_ae", loss_ae, prog_bar=True)
        self.log("train_loss_adv", loss_adv_classifier, prog_bar=True)
        self.log("train_cluster_loss", loss_cluster, prog_bar=True)
        self.log("grl_lambda", grl_lambda, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # For validation, use the default GRL strength stored in the model.
        output = self.model(batch, lambda_adv=self.model.lambda_adv)
        loss_recon = self._compute_recon_loss(batch, output['reconstruction'])
        loss_adv = self.class_loss_fn(output['turbine_prediction'], batch['turbine_name'])
        total_loss = loss_recon + self.adv_loss_weight * loss_adv
        self.log("val_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        # Use a single optimizer for all parameters.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_ae)
        lr_scheduler_config = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler_config]
