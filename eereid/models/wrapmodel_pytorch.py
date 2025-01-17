from eereid.models.model import model
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class extending_layer(nn.Module):
    def forward(self, inputs):
        return inputs.unsqueeze(0)

class concat_layer(nn.Module):
    def forward(self, inputs):
        return torch.cat(inputs, dim=0)

class wrapmodel_pytorch(model):
    def __init__(self,name):
        super().__init__("wrap_"+name)

    def build_submodel(self, input_shape, mods):
        raise NotImplementedError

    # TODO this needs rethinking, model was not built as it should
    def build(self, input_shape, siamese_count, mods):
        self.build_submodel(input_shape, mods)

        self.siamese_count = siamese_count
        self.input_shape = input_shape

        self.extending_layer = extending_layer()
        self.concat_layer = concat_layer()
        

    def forward(self, x):
        samples = [x[:, i] for i in range(self.siamese_count)]
        samples = [self.submodel(sample) for sample in samples]
        samples = [self.extending_layer(sample) for sample in samples]
        outp = self.concat_layer(samples)
        return outp

    def explain(self):
        return "Inheritable generic class creating siamese neural network wrapper"
    
    def summary(self,*args,**kwargs):
        print("submodel:")
        summary(self.submodel,tuple([self.input_shape[2],self.input_shape[0],self.input_shape[1]]))
        print("model:")
        summary(self.model,tuple([self.input_shape[2],self.input_shape[0],self.input_shape[1]]))
