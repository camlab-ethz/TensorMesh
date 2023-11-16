import re
import torch
import torch.nn as nn
from itertools import chain
class BufferDict(nn.Module):
    def __init__(self, data = None):
        super().__init__()
        if data is None:
            data = {}
        self._data = {} # used for storing data that cannot be used as a valid name
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
      
        for key in list(data.keys()):
            if not pattern.match(key):
                self._data[key] = data.pop(key)
        for key, value in data.items():
            if isinstance(value, nn.Module):
                setattr(self, key, value)
            elif isinstance(value, torch.Tensor):
                self.register_buffer(key, value)
            else:
                raise TypeError(f"Cannot register a {type(value)} as a buffer or a parameter")
    
    def as_parameter(self, key):
        buffer = self._buffers.pop(key)
        self.register_parameter(key, buffer)
        
    def as_buffer(self, key):
        parameter = self._parameters.pop(key)
        self.register_buffer(key, parameter)
        
    def keys(self):
        return chain(self._buffers.keys(), self._parameters.keys(), self._data.keys(), self._modules.keys())
    
    def items(self):
        return chain(self._buffers.items(), self._parameters.items(), self._data.items(), self._modules.items())
    
    def values(self):
        return chain(self._buffers.values(), self._parameters.values(), self._data.values(), self._modules.values())
    
    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(f"{key} is not found in the BufferDict")
        
        return self._buffers[key] if key in self._buffers else self._parameters[key] if key in self._parameters else self._data[key] if key in self._data else self._modules[key]

    def __setitem__(self, key, value):
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
        if not pattern.match(key):
            self._data[key] = value
        else:
            if isinstance(value, nn.Module):
                setattr(self, key, value)
            elif isinstance(value, torch.Tensor):
                self.register_buffer(key, value)
            else:
                raise TypeError(f"Cannot register a {type(value)} as a buffer or a parameter")

    def __len__(self):
        return len(self._buffers) + len(self._parameters) + len(self._data)
    
    def __includes__(self, key):
        return key in self.keys()

    @property
    def dtype(self):
        return next(iter(self.buffers().values())).dtype


    @property
    def device(self):
        return next(iter(self.buffers().values())).device
    
    def _apply(self, fn):
        self = super()._apply(fn)
        for key, value in self._data.items():
            self._data[key] = fn(value)
        return self

    def __str__(self):
        return f"""BufferDict(
        {', '.join([f"{key} = {value}" for key, value in self.items()])}
        )"""
      
    def __repr__(self):
        return str(self)