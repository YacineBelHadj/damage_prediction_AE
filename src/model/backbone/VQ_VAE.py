import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.model.backbone.utils import build_layers
from src.model.transformation import StandardScaler

# Assuming build_layers and StandardScaler are defined elsewhere
# from your_module import build_layers, StandardScaler
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Vector Quantizer as per VQ-VAE paper.

        Args:
            num_embeddings (int): Number of vectors in the codebook.
            embedding_dim (int): Dimensionality of each embedding vector.
            commitment_cost (float): Weight for the commitment loss.
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        """
        Connects the encoder and decoder through vector quantization.

        Args:
            z (Tensor): Encoder output of shape (batch, embedding_dim) or (batch, embedding_dim, H, W).

        Returns:
            quantized (Tensor): Quantized latent vectors.
            loss (Tensor): VQ loss (codebook + commitment).
            perplexity (Tensor): Perplexity of the quantizer usage.
            encodings (Tensor): One-hot encodings of the nearest embeddings.
        """
        # Handle 2D and 4D tensors
        if z.dim() == 2:
            z_flat = z  # Shape: (batch, embedding_dim)
            batch_size = z.size(0)
        elif z.dim() == 4:
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)  # Shape: (batch*H*W, embedding_dim)
            batch_size = z.size(0)
        else:
            raise ValueError(f"Unsupported z dimensions: {z.dim()}")

        # Calculate distances between z and embeddings
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )  # Shape: (batch*H*W, num_embeddings) or (batch, num_embeddings)

        # Find the nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # Shape: (batch*H*W, 1) or (batch, 1)

        # Convert to one-hot encodings
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)  # Shape: (batch*H*W, num_embeddings) or (batch, num_embeddings)

        # Quantize the latent vectors
        quantized = torch.matmul(encodings, self.embeddings.weight)  # Shape: (batch*H*W, embedding_dim) or (batch, embedding_dim)

        # Compute VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), z_flat)
        q_latent_loss = F.mse_loss(quantized, z_flat.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Compute perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Use the straight-through estimator
        quantized = z_flat + (quantized - z_flat).detach()

        # Reshape quantized to original shape
        if z.dim() == 4:
            quantized = quantized.view(batch_size, self.embedding_dim, z.size(2), z.size(3))  # Shape: (batch, embedding_dim, H, W)

        return quantized, loss, perplexity, encodings
class MultiHeadVQVAE(nn.Module):
    def __init__(self, input_spec: dict = {'Welch_X': 264,
                                         'Welch_Y': 264,
                                         'Welch_Z': 264},
                 latent_dim: int = 32,  # Should match shared_layer[-1]
                 num_embeddings: int = 512,
                 commitment_cost: float = 0.25,
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

        self.turbine_embedding = nn.Embedding(44, 16)  # Adjusted embedding dimension

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
        shared_input_dim = len(input_spec) * separated_layer[-1]  # e.g., 3 * 32 = 96
        self.encoder_shared = build_layers(
            hidden_dims=[shared_input_dim] + shared_layer,  # e.g., [96, 128, 64, 32]
            activation_list=activation_str,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            norm_layer=None,  # Assuming normalization is handled in separate encoders
        )

        # Vector Quantizer
        self.vq_layer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=shared_layer[-1],  # e.g., 32
            commitment_cost=commitment_cost
        )

        # Shared decoder layers before separation
        shared_layer_decoder = [latent_dim] + shared_layer[::-1]  # e.g., [32, 32, 64, 128]
        separated_layer_decoder = [shared_layer_decoder[-1] + 16] + separated_layer[::-1]  # 16 from turbine_embedding

        self.decoder_shared = build_layers(
            hidden_dims=shared_layer_decoder,
            activation_list=activation_str,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            norm_layer=None,
        )

        # Separate decoders for each output
        self.decoder_separated = nn.ModuleDict({
            name: build_layers(
                hidden_dims=separated_layer_decoder + [dim],
                activation_list=['relu'] * len(separated_layer_decoder) + [None],  # No activation for last layer
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
        Encodes each input separately, concatenates them, and passes through shared layers.
        """
        # Encode each input separately
        encoded_separated = []
        for name, encoder in self.encoder_separated.items():
            encoded = encoder(x[name])  # Shape: (batch, separated_layer[-1])
            encoded_separated.append(encoded)

        # Concatenate all encoded features
        concatenated = torch.cat(encoded_separated, dim=1)  # Shape: (batch, shared_input_dim)

        # Pass through shared encoder layers
        shared_encoded = self.encoder_shared(concatenated)  # Shape: (batch, shared_layer[-1])

        return shared_encoded

    def decode(self, quantized, turbine_id):
        """
        Decodes the quantized latent vectors into each output modality.
        """
        # Pass through shared decoder layers
        shared_decoded = self.decoder_shared(quantized)  # Shape: (batch, shared_layer_decoder[-1])

        # Embed turbine IDs and concatenate
        turb_emb = self.turbine_embedding(turbine_id)  # Shape: (batch, 16)
        shared_decoded_cond = torch.cat([shared_decoded, turb_emb], dim=1)  # Shape: (batch, shared_layer_decoder[-1] + 16)

        # Decode each modality separately
        decoded = {}
        for name, decoder in self.decoder_separated.items():
            decoded[name] = decoder(shared_decoded_cond)  # Shape: (batch, dim)

        return decoded

    def forward(self, x: dict):
        """
        Forward pass through the VQ-VAE.
        """
        shared_encoded = self.encode(x)  # Shape: (batch, shared_layer[-1])
        quantized, vq_loss, perplexity, _ = self.vq_layer(shared_encoded)  # quantized shape: (batch, shared_layer[-1])
        reconstructed = self.decode(quantized, x['turbine_name'])  # Shape: (batch, dim) for each modality

        # Construct output dictionary including latent vectors
        output = {
            'reconstruction': reconstructed,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'latent_vectors': quantized  # Add quantized latent vectors
        }

        return output
class TrainingMultiHeadVQVAE(pl.LightningModule):
    def __init__(self, input_spec: dict = {'Welch_X': 264,
                                         'Welch_Y': 264,
                                         'Welch_Z': 264},
                 latent_dim: int = 64,
                 num_embeddings: int = 512,
                 commitment_cost: float = 0.25,
                 separated_layer: list[int] = [128, 64, 32],
                 shared_layer: list[int] = [128, 64, 32],
                 activation_str: str = 'relu',
                 batch_norm: bool = True,
                 dropout_rate: float = 0.0,
                 normalization_values: dict = None,
                 lr: float = 1e-3,
                 recon_weight: float = 1.0,
                 vq_weight: float = 1.0):
        """
        PyTorch Lightning module for training the Multi-Head VQ-VAE.

        Args:
            input_spec (dict): Dictionary with input names as keys and their dimensions as values.
            latent_dim (int): Dimension of the latent space.
            num_embeddings (int): Number of embeddings in the VQ codebook.
            commitment_cost (float): Weight for the commitment loss in VQ.
            separated_layer (list[int]): Hidden layer dimensions for separated encoders/decoders.
            shared_layer (list[int]): Hidden layer dimensions for shared encoders/decoders.
            activation_str (str): Activation function to use.
            batch_norm (bool): Whether to use batch normalization.
            dropout_rate (float): Dropout rate.
            normalization_values (dict): Normalization parameters for each input.
            lr (float): Learning rate.
            recon_weight (float): Weight for the reconstruction loss.
            vq_weight (float): Weight for the vector quantization loss.
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_spec = input_spec
        self.model = MultiHeadVQVAE(
            input_spec=input_spec,
            latent_dim=latent_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            separated_layer=separated_layer,
            shared_layer=shared_layer,
            activation_str=activation_str,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            normalization_values=normalization_values
        )

        self.lr = lr
        self.recon_weight = recon_weight
        self.vq_weight = vq_weight
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (dict): Dictionary containing input tensors for each modality.

        Returns:
            dict: Dictionary containing reconstructions, VQ loss, and perplexity.
        """
        return self.model(x)

    def _compute_loss(self, x, output):
        """
        Computes the combined reconstruction and VQ losses.

        Args:
            x (dict): Original inputs.
            output (dict): Output from the model containing 'reconstruction', 'vq_loss', and 'perplexity'.

        Returns:
            Tensor: Combined loss.
        """
        # Reconstruction loss: sum MSE loss for each modality
        recon_loss = 0
        for name in self.input_spec.keys():
            recon_loss += self.loss_fn(output['reconstruction'][name], x[name])

        # VQ Loss
        vq_loss = output['vq_loss']

        # Total loss
        total_loss = self.recon_weight * recon_loss + self.vq_weight * vq_loss
        return total_loss, recon_loss, vq_loss, output['perplexity']

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
        loss, recon_loss, vq_loss, perplexity = self._compute_loss(batch, output)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_vq_loss", vq_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True)
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
        loss, recon_loss, vq_loss, perplexity = self._compute_loss(batch, output)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_vq_loss", vq_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True)
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
