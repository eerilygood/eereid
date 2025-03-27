from eereid.losses.loss import loss

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    from eereid.importhelper import importhelper
    torch=importhelper("torch","wrapmodel_pytorch","pip install torch")

class quadruplet(loss):
    def __init__(self, margin=1.0):
        self.margin = margin
        super().__init__("quadruplet")

    def build(self,mods):

        typ=mods("loss_aggregator","avg")

        def func(y_true, y_pred,use_pytorch=False):
            if not use_pytorch:
                anchor, positive, negative, negative2 = y_pred[0], y_pred[1], y_pred[2], y_pred[3]
                positive_dist = K.sum(K.square(anchor - positive), axis=-1)
                negative_dist = K.sum(K.square(anchor - negative), axis=-1)
                negative_dist2= K.sum(K.square(anchor - negative2), axis=-1)
                if typ=="min":
                    return K.sum(K.maximum(positive_dist - K.minimum(negative_dist,negative_dist2)  + self.margin, 0), axis=-1)
                elif typ=="max":
                    return K.sum(K.maximum(positive_dist - K.maximum(negative_dist,negative_dist2)  + self.margin, 0), axis=-1)
                elif typ=="avg":
                    return K.sum(K.maximum(positive_dist - (negative_dist+negative_dist2)/2  + self.margin, 0), axis=-1)
                elif typ=="sum":
                    return K.sum(K.maximum(positive_dist - (negative_dist+negative_dist2)  + self.margin, 0), axis=-1)
                else:
                    raise ValueError("Invalid type",typ)
            else:
                anchor, positive, negative, negative2 = y_pred[0], y_pred[1], y_pred[2], y_pred[3]
                positive_dist = torch.sum((anchor - positive) ** 2, dim=-1)
                negative_dist = torch.sum((anchor - negative) ** 2, dim=-1)
                negative_dist2 = torch.sum((anchor - negative2) ** 2, dim=-1)
                if typ=="min":
                    return torch.sum(F.relu(positive_dist - torch.minimum(negative_dist,negative_dist2)  + self.margin),dim=-1)
                elif typ=="max":
                    return torch.sum(F.relu(positive_dist - torch.maximum(negative_dist,negative_dist2)  + self.margin),dim=-1)
                elif typ=="avg":
                    return torch.sum(F.relu(positive_dist - (negative_dist+negative_dist2)/2  + self.margin),dim=-1)
                elif typ=="sum":
                    return torch.sum(F.relu(positive_dist - (negative_dist+negative_dist2)  + self.margin),dim=-1)
                else:
                    raise ValueError("Invalid type",typ)

        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)
        
    def Nlet_string(self):
        return "aabc"

    def explain(self):
        return "Quadruplet loss with margin of "+str(self.margin)+". The formula is relu(D(a,p)-loss_aggregator(D(a,n),D(a,n2))+margin)."

