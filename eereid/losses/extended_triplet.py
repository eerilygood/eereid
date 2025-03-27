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

class extended_triplet(loss):
    def __init__(self, margin=1.0):
        self.margin = margin
        super().__init__("triplet")

    def build(self,mods):

        typ=mods("loss_aggregator","avg")

        def func(y_true, y_pred,use_pytorch=False):
            anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
            if not use_pytorch:
                positive_dist = K.sum(K.square(anchor - positive), axis=-1)
                negative_dist = K.sum(K.square(anchor - negative), axis=-1)
                both_dist     = K.sum(K.square(positive - negative), axis=-1)
                if typ=="min":#bad choice?
                    return K.mean(K.maximum(positive_dist - K.minimum(negative_dist,both_dist)  + self.margin, 0), axis=-1)
                elif typ=="max":#seems good
                    return K.mean(K.maximum(positive_dist - K.maximum(negative_dist,both_dist)  + self.margin, 0), axis=-1)
                elif typ=="avg":#maybe even better
                    return K.mean(K.maximum(positive_dist - (negative_dist+both_dist)/2  + self.margin, 0), axis=-1)
                elif typ=="sum":#generally a bad choice because it is not scale invariant
                    return K.mean(K.maximum(positive_dist - (negative_dist+both_dist)  + self.margin, 0), axis=-1)
                else:
                    raise ValueError("Invalid type",typ)
            else:
                positive_dist = torch.sum((anchor - positive) ** 2, dim=-1)
                negative_dist = torch.sum((anchor - negative) ** 2, dim=-1)
                both_dist     = torch.sum((positive - negative) ** 2, dim=-1)
                if typ=="min":
                    return torch.mean(F.relu(positive_dist - torch.minimum(negative_dist,both_dist)  + self.margin),dim=-1)
                elif typ=="max":
                    return torch.mean(F.relu(positive_dist - torch.maximum(negative_dist,both_dist)  + self.margin),dim=-1)
                elif typ=="avg":
                    return torch.mean(F.relu(positive_dist - (negative_dist+both_dist)/2  + self.margin),dim=-1)
                elif typ=="sum":
                    return torch.mean(F.relu(positive_dist - (negative_dist+both_dist)  + self.margin),dim=-1)
                else:
                    raise ValueError("Invalid type",typ)

        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)
        
    def Nlet_string(self):
        return "aab"

    def explain(self):
        return "Extended triplet loss with margin of "+str(self.margin)+". The formula is relu(D(a,p)-loss_aggregator(D(a,n),D(p,n))+margin)."

