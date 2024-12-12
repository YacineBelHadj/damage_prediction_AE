import torch 
from torch import nn
from src.model.transformation import NormLayer, UnsqueezeLayer, SqueezeLayer

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
    
    
    
class AutoEncoderConv(nn.Module):
    def __init__(self):
        super(AutoEncoderConv, self).__init__()
        
        self.encoder = nn.Sequential(
            NormLayer(-22.88, -2.31),
            UnsqueezeLayer(1),
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=253, stride=1, padding=1),
            SqueezeLayer(-1),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 263),
            NormLayer(-22.88, -2.31, denormalize=True)
        )

        # Register buffers for tracking reconstruction errors
        self.register_buffer('rec_err', torch.tensor(float('inf')))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def forward_embedding(self, x):
        return self.encoder(x)
    
    def update_rec_err(self, rec_err):
        self.rec_err = torch.min(self.rec_err, rec_err)
        
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
        