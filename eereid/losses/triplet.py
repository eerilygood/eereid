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
    def __init__(self, margin=1.0,pytorch=False):
        self.margin = margin
        self.pytorch=pytorch
        super().__init__("triplet")

    def build(self,mods):
        if not self.pytorch:
            def func(y_true, y_pred):
                # print(y_true.shape,y_pred.shape)
                #exit()
                anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
                # print("anchor.shape: ",anchor.shape)
                # print("positive.shape: ",positive.shape)
                # print("negative.shape: ",negative.shape)
                positive_dist = K.sum(K.square(anchor - positive), axis=-1)
                negative_dist = K.sum(K.square(anchor - negative), axis=-1)
                # print("positive_dist.shape: ",positive_dist.shape)
                # print("negative_dist.shape: ",negative_dist.shape)
                return K.mean(K.maximum(positive_dist - negative_dist + self.margin, 0), axis=-1)
        else:
            def func(y_true, y_pred):
                # print(y_true.shape,y_pred.shape)
                anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
                # print("anchor.shape: ",anchor.shape)
                # print("positive.shape: ",positive.shape)
                # print("negative.shape: ",negative.shape)
                positive_dist = torch.sum((anchor - positive) ** 2, dim=-1)
                negative_dist = torch.sum((anchor - negative) ** 2, dim=-1)
                # print("positive_dist.shape: ",positive_dist.shape)
                # print("negative_dist.shape: ",negative_dist.shape)

                return torch.mean(F.relu(positive_dist - negative_dist + self.margin),dim=-1)
        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)

    def Nlet_string(self):
        return "aab"

    def explain(self):
        return "Triplet loss with margin of "+str(self.margin)+". The formula is relu(D(a,p)-D(a,n)+margin)."
        

