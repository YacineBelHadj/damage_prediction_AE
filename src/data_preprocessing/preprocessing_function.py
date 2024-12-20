from torchaudio.transforms import Spectrogram, Resample
import torch
from torch import nn

class Welch(nn.Module):
    def __init__(self,n_fft=4096,average='mean'):
        super(Welch, self).__init__()
        self.n_fft = n_fft
        self.spectrogram = Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=2048)
        self.average = average
        self.average_dict = {'mean': torch.mean, 'std': torch.std}
        if average not in self.average_dict.keys():
            raise ValueError(f"average must be one of {self.average.keys()}")
        self.average_fn = self.average_dict[average]
        
    def get_frequency_axis(self, sample_rate):
        return torch.linspace(0, sample_rate / 2, int(self.n_fft // 2) + 1)
        
    def forward(self, x):
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # detrend
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return self.average_fn(self.spectrogram(x), dim=-1)
    def __str__(self):
        return f"Welch(n_fft={self.n_fft}, average={self.average})"
    
class DownSample(nn.Module):
    def __init__(self, fs_in: float, fs_out: float,**kwargs):
        super().__init__()
        if fs_out >= fs_in:
            raise ValueError("Output sampling rate must be less than input sampling rate")
        self.fs_in = fs_in
        self.fs_out = fs_out
        self.resample = Resample(fs_in,fs_out,**kwargs)
        
    def forward(self,x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.resample(x)
    
class RollingAverage(nn.Module):
    def __init__(self, window_size:int,stride_ratio:float=0.5):
        super(RollingAverage, self).__init__()
        self.window_size = window_size
        self.stride_ratio = stride_ratio
    def forward(self, x):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x =  torch.nn.functional.avg_pool1d(x.unsqueeze(0), kernel_size=self.window_size, stride=int(self.window_size*self.stride_ratio)).squeeze(0)
        return x 
    def __str__(self):
        # return a description of the module and its parameters
        return f"RollingAverage(window_size={self.window_size}, stride_ratio={self.stride_ratio})"
    
class RMS(nn.Module):
    def __init__(self):
        super(RMS, self).__init__()
    def forward(self, x):
        x_detrend = x - torch.mean(x, dim=-1, keepdim=True)
        return torch.sqrt(torch.mean(x_detrend**2, dim=-1))
    def __str__(self):
        return f"RMS"
    
class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()
    def forward(self, x):
        return torch.mean(x, dim=-1)
    def __str__(self):
        return f"Mean"

    
class Range(nn.Module):
    def __init__(self):
        super(Range, self).__init__()
    def forward(self, x):
        return torch.max(x, dim=-1).values - torch.min(x, dim=-1).values
    def __str__(self):
        return f"Range"
    
class ParallelModule(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.module_dict = {modu.__class__.__name__: modu for modu in modules}
    
    def forward(self, x):
        # Apply all modules in parallel
        return {name: module(x) for name, module in self.module_dict.items()}
    
    def __str__(self):
        module_descriptions = ' '.join(
            [f"{name}: {str(module)}" for name, module in self.module_dict.items()]
        )
        return f"ParallelModule(\n{module_descriptions}\n)"
