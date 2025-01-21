from torch import nn
from src.model.backbone.utils import build_conv_layers

class ConvAutoEncoder(nn.Module):
    """
    A 1D Convolutional AutoEncoder that uses build_conv_layers for both:
      - An encoder (forward conv)
      - A decoder (transpose conv)

    We impose a latent_dim by flattening the encoder's final output
    and mapping it (linear) -> latent_dim, then reversing that for the decoder.
    """
    def __init__(
        self,
        encoder_specs,       # e.g. [(3,16,5,2),(16,32,3,2)]
        decoder_specs,       # e.g. [(32,16,4,2),(16,3,4,2)]
        latent_dim=32,
        input_dim=264,
        encoder_activation=None,
        decoder_activation=None,
        batch_norm=True,
        dropout_rate=0.0,
        debug=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # 1) Build encoder
        self.encoder, (enc_ch, enc_len) = build_conv_layers(
            conv_specs=encoder_specs,
            activation_list=encoder_activation,  # e.g. 'relu' or ['relu','relu']
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            debug=debug,
            input_dim=input_dim,
            convtranspose=False  # standard conv
        )
        if enc_ch is None or enc_len is None:
            raise ValueError("Encoder specs produced invalid shape (None).")

        # We'll flatten from (enc_ch, enc_len) => enc_ch*enc_len
        enc_out_dim = enc_ch * enc_len
        self.enc_ch = enc_ch
        self.enc_len = enc_len

        # Linear from enc_out_dim -> latent_dim
        self.linear_enc = nn.Linear(enc_out_dim, latent_dim)
        
        # 2) Build decoder
        # We'll *start* from shape (enc_ch, enc_len) for the transpose conv.
        self.decoder, (dec_ch, dec_len) = build_conv_layers(
            conv_specs=decoder_specs,
            activation_list=decoder_activation,  # e.g. 'relu'
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            debug=debug,
            input_dim=enc_len,
            convtranspose=True  # use transpose conv
        )
        # We do a linear from latent_dim -> enc_out_dim
        self.linear_dec = nn.Linear(latent_dim, enc_out_dim)

        self.dec_ch = dec_ch
        self.dec_len = dec_len
        # Note: hopefully dec_ch == original in_channels and dec_len == input_dim if you mirrored.

    def encode(self, x):
        """
        x shape: (B, in_channels, length) 
                 or if using a dictionary approach, your encoder might 
                 handle that via norm_layer+DictStack. 
        """
        # Pass through encoder conv
        out = self.encoder(x)     # shape (B, enc_ch, enc_len)
        B, C, L = out.shape
        x_flat = out.view(B, -1)  # (B, C*L)
        latent = self.linear_enc(x_flat)  # (B, latent_dim)
        return latent

    def decode(self, z):
        """
        z shape: (B, latent_dim).
        """
        B = z.size(0)
        # Expand back to (B, enc_ch * enc_len)
        rec_flat = self.linear_dec(z)            # (B, enc_ch*enc_len)
        rec_unflat = rec_flat.view(B, self.enc_ch, self.enc_len)
        # Pass through transpose conv
        out = self.decoder(rec_unflat)           # e.g. (B, out_channels, final_length)
        return out

    def forward(self, x):
        lat = self.encode(x)
        recon = self.decode(lat)
        return recon, lat