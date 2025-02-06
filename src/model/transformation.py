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
        self.freq_axis_masked = self.freq_axis[self.freq_mask]
        
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
    
import torch
from torch.nn import Module

class EncoderBasedOnList(Module):
    def __init__(self, list_of_elements: list = None, encode: bool = True):
        super().__init__()
        
        if list_of_elements is None or not isinstance(list_of_elements, list):
            raise ValueError("list_of_elements must be a non-empty list.")
        
        self.encode = encode
        self.list_of_elements = list_of_elements
        self.encode_map = {element: i for i, element in enumerate(self.list_of_elements)}
        self.decode_map = {v: k for k, v in self.encode_map.items()}
        
    def fw_encode(self, element):
        if element not in self.encode_map:
            raise ValueError(f"Element '{element}' not found in encode_map.")
        return self.encode_map[element]

    def fw_decode(self, element):
        if element not in self.decode_map:
            raise ValueError(f"Element '{element}' not found in decode_map.")
        return self.decode_map[element]

    def forward(self, element):
        # Choose the correct function based on self.encode
        func = self.fw_encode if self.encode else self.fw_decode
        
        # Handle batched inputs (e.g., lists or tensors)
        if isinstance(element, (list, torch.Tensor,np.ndarray)):
            return torch.tensor([func(e) for e in element])
        else:
            # Handle single element
            return torch.tensor(func(element))


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
        
class StandardScaler(Module):
    def __init__(self, mean=None, std=None, denormalize=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.denormalize = denormalize
        self.forward_func = {False: self.forward_norm, True: self.forward_denorm}
        self.forward = self.forward_func[self.denormalize]

    def forward_norm(self, x):
        return (x - self.mean) / (self.std )

    def forward_denorm(self, x):
        return x * (self.std) + self.mean
    
    def forward(self, x):
        return self.forward(x)
        
class UnsqueezeLayer(Module):
    def __init__(self, dim):
        super(UnsqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        if x.dim() > self.dim:
            return x
        return x.unsqueeze(self.dim)

class SqueezeLayer(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze(self.dim)
        return x
    
    
from torch.nn import Module, ModuleDict
import torch

class MultiChannelNormalization(Module):
    def __init__(self, sensor_list: list, normalization_type: str,statistics:dict, denormalize: bool = False):
        super().__init__()
        self.sensor_list = sensor_list
        self.normalization_type = normalization_type
        self.statistics = statistics
        self.denormalize = denormalize

        # We'll create a sub-module for each key in input_spec
        self.norm_layers = ModuleDict()

        for channel in sensor_list:
            # e.g., params = { 'min_values': -22, 'max_values': 7 }
            # or       = { 'mean': 0.1, 'std': 1.2 }, depending on the type
            # We pass these to the normalization class along with `denormalize`.

            # For NormLayer, signature might be:
            #   NormLayer(max_val, min_val, denormalize=bool)
            # For StandardScaler, signature might be:
            #   StandardScaler(mean, std, denormalize=bool)

            # We'll detect the relevant keys:
            if self.normalization_type == 'minmax':
                # Expect 'max_values' and 'min_values'
                max_val = statistics[channel]['max']
                min_val = statistics[channel]['min']
                layer = NormLayer(max_val, min_val, denormalize=denormalize)
            elif self.normalization_type == 'meanstd':
                # Expect 'mean' and 'std'
                mean_val = statistics[channel]['mean']
                std_val = statistics[channel]['std']
                layer = StandardScaler(mean_val, std_val, denormalize=denormalize)
            else:
                raise ValueError(f"Unsupported normalization_type: {normalization_type}")

            self.norm_layers[channel] = layer

    def forward(self, x_dict: dict) -> dict:
        """
        Applies the per-key normalization to each entry in x_dict.
        Example:
            x_dict = {
                'Welch_X': torch.tensor([...]),
                'Welch_Y': torch.tensor([...]),
                ...
            }
        Returns a new dict with normalized (or denormalized) tensors.
        """
        output = {}
        for sensor in self.sensor_list:
            
            # Apply the corresponding norm layer
            norm_layer = self.norm_layers[sensor]
            output[sensor] = norm_layer(x_dict[sensor])
        return output
import torch
from torch import nn

class DictStack(nn.Module):
    def __init__(self, keys, dim=1, unstack=False):
        """
        Args:
            keys (list): List of keys to stack from the input dictionary.
            dim (int): Dimension along which to stack the tensors.
            unstack (bool): If True, unstack the tensor back into a dictionary.
        """
        super().__init__()
        self.keys = keys
        self.dim = dim
        self.unstack = unstack

    def forward(self, x):
        """
        Args:
            x (dict): A dictionary of tensors.

        Returns:
            If unstack is False, returns a stacked tensor.
            If unstack is True, returns a dictionary of unstacked tensors.
        """
        if self.unstack:
            # Unstack the tensor into a dictionary
            if not isinstance(x, torch.Tensor):
                raise ValueError("Input must be a tensor when unstack is True.")
            tensors = torch.unbind(x, dim=self.dim)
            if len(tensors) != len(self.keys):
                raise ValueError(f"Number of tensors ({len(tensors)}) does not match number of keys ({len(self.keys)}).")
            return {key: tensor for key, tensor in zip(self.keys, tensors)}
        else:
            # Stack tensors from the dictionary
            if not isinstance(x, dict):
                raise ValueError("Input must be a dictionary when unstack is False.")
            tensors = [x[key] for key in self.keys]
            return torch.stack(tensors, dim=self.dim)
        
class CutDict(nn.Module):
    def __init__(self, input_spec:dict):
        super().__init__()
        self.input_spec = input_spec
    def forward(self, x):
        """cut the input tensor to the specified lengths"""
        return {key: x[key][...,:self.input_spec[key]] for key in self.input_spec.keys()}
    
    
if __name__ == '__main__':
    # Example usage
    sensor_list = ['Welch_X', 'Welch_Y', 'Welch_Z']
    statistics = {'Welch_X': {'max': 7, 'min': -22},
                  'Welch_Y': {'max': 7, 'min': -18},
                    'Welch_Z': {'max': 8, 'min': -16}}
    
    multi_norm = MultiChannelNormalization(sensor_list, NormLayer, statistics, denormalize=False)
    # Example input data
    x_dict = {
        'Welch_X': torch.tensor([-22, -21, -20, -19, -18, 7], dtype=torch.float),
        'Welch_Y': torch.tensor([-18, -17, -16, -15, -14, 7], dtype=torch.float),
        'Welch_Z': torch.tensor([-16, -15, -14, -13, -12, 8], dtype=torch.float),
    }

    # Normalize the data
    x_norm = multi_norm(x_dict)
    print(x_norm)
    print(type(x_norm))
    multi_norm = MultiChannelNormalization(sensor_list, NormLayer, statistics,denormalize=True)

    # Denormalize the data
    x_denorm = multi_norm(x_norm)
    print(x_denorm)
    
    assert torch.allclose(x_dict['Welch_X'], x_denorm['Welch_X'])
    
