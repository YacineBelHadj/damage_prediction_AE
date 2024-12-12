from torch.nn import Module
import torch
import numpy as np

class ToTensor(Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return torch.tensor(x, dtype=self.dtype)

class CutPSD(Module):
    def __init__(self, freq_axis:torch.Tensor | np.ndarray, freq_range:tuple[int, int]):
        super().__init__()
        self.freq_axis = freq_axis  
        self.freq_range = freq_range
        self.freq_mask = (self.freq_axis >= self.freq_range[0]) & (self.freq_axis <= self.freq_range[1])
        
    def forward(self, psd:torch.Tensor):
        if psd.ndim == 1:
            return psd[self.freq_mask]
        return psd[:,self.freq_mask]
    
class FromBuffer(Module):
    def __init__(self, dtype=np.float32):
        super().__init__()
        self.dtype = dtype

    def forward(self, buffer):
        array = np.frombuffer(buffer, dtype=self.dtype)
        array = np.copy(array)  # Make the array writable
        return torch.tensor(array)
    
class LogTransform(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.log(x)
    
class EncoderBasedOnList(Module):
    def __init__(self, list_of_elements: list = None, encode: bool = True):
        super().__init__()
        self.encode = encode
        self.list_of_elements = list_of_elements
        self.forward_func = {True: self.fw_encode, False: self.fw_decode}
        self.encode_map = {element: i for i, element in enumerate(self.list_of_elements)}
        self.decode_map = {v: k for k, v in self.encode_map.items()}
        
        def fw_encode(self, element):
            return self.encode_map[element]
        def fw_decode(self, element):
            return self.decode_map[element]
        def forward(self, element):
            return torch.tensor(self.forward_func[self.encode](element))

class NormLayer(Module):
    def __init__(self, max_val, min_val, denormalize=False,dtypes=torch.float32):
        super().__init__()
        self.dtypes = dtypes
        self.register_buffer('max', self._to_tensor(max_val))
        self.register_buffer('min', self._to_tensor(min_val))
        self.denormalize = denormalize
        self.forward_func = {False: self.forward_norm, True: self.forward_denorm}
        self.forward = self.forward_func[self.denormalize]

    def forward_norm(self, x):
        return (x - self.min) / (self.max - self.min + 1e-8)

    def forward_denorm(self, x):
        return x * (self.max - self.min + 1e-8) + self.min

    def _to_tensor(self, val):
        if isinstance(val, torch.Tensor):
            return val.clone().detach()
        else:
            return torch.tensor(val, dtype=self.dtypes)
        
        
class UnsqueezeLayer(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.unsqueeze(self.dim)
        return x

class SqueezeLayer(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze(self.dim)
        return x