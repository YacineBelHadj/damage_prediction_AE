from torch import nn
import torch
import numpy as np


class CutPSD(nn.Module):
    def __init__(self, freq_axis:torch.Tensor | np.ndarray, freq_range:tuple[int, int]):
        super(CutPSD, self).__init__()
        self.freq_axis = freq_axis  
        self.freq_range = freq_range
        self.freq_mask = (self.freq_axis >= self.freq_range[0]) & (self.freq_axis <= self.freq_range[1])
        
    def forward(self, psd:torch.Tensor):
        return psd[self.freq_mask]
    
class FromBuffer(nn.Module):
    def __init__(self, dtype=np.float32):
        super(FromBuffer, self).__init__()
        self.dtype = dtype

    def forward(self, buffer):
        array = np.frombuffer(buffer, dtype=self.dtype)
        array = np.copy(array)  # Make the array writable
        return torch.tensor(array)
    
class EncoderBasedOnList(nn.Module):
    def __init__(self, list_of_elements: list = None, encode: bool = True):
        super(EncoderBasedOnList, self).__init__()
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

class NormLayer(nn.Module):
    def __init__(self, max_val, min_val, denormalize=False):
        super(NormLayer, self).__init__()
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
            return torch.tensor(val, dtype=torch.float32)
        
        
        