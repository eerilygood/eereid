from eereid.losses.loss import loss

import numpy as np

from tensorflow.keras import backend as K
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    from eereid.importhelper import importhelper
    torch=importhelper("torch","wrapmodel_pytorch","pip install torch")

class triplet(loss):
    def __init__(self, margin=1.0):
        self.margin = margin
        # self.pytorch=pytorch
        super().__init__("triplet")

    def build(self,mods):   

        def func(y_true, y_pred,use_pytorch=False):
            if not use_pytorch:

                anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
                positive_dist = K.sum(K.square(anchor - positive), axis=-1)
                negative_dist = K.sum(K.square(anchor - negative), axis=-1)

                return K.mean(K.maximum(positive_dist - negative_dist + self.margin, 0), axis=-1)
            else:

                anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
                positive_dist = torch.sum((anchor - positive) ** 2, dim=-1)
                negative_dist = torch.sum((anchor - negative) ** 2, dim=-1)

                return torch.mean(F.relu(positive_dist - negative_dist + self.margin),dim=-1)
        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)

    def Nlet_string(self):
        return "aab"

    def explain(self):
        return "Triplet loss with margin of "+str(self.margin)+". The formula is relu(D(a,p)-D(a,n)+margin)."
        

