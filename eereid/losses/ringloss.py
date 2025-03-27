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

class ringloss(loss):
    def __init__(self, margin=10.0):
        self.margin = margin
        super().__init__("ringloss")

    def build(self,mods):

        typ=mods("loss_aggregator","avg")

        def func(y_true, y_pred,use_pytorch=False):
            if not use_pytorch:
                a,b,c=y_pred[0],y_pred[1],y_pred[2]
                d1=K.sum(K.square(a-b),axis=-1)
                d2=K.sum(K.square(b-c),axis=-1)
                d3=K.sum(K.square(c-a),axis=-1)
                aa=K.sum(K.square(a),axis=-1)
                bb=K.sum(K.square(b),axis=-1)
                cc=K.sum(K.square(c),axis=-1)

                loss=K.maximum(0.0,self.margin+aa+bb+cc-d1-d2-d3)
                return loss
            else:
                a,b,c=y_pred[0],y_pred[1],y_pred[2]
                d1=torch.sum((a-b)**2,dim=-1)
                d2=torch.sum((b-c)**2,dim=-1)
                d3=torch.sum((c-a)**2,dim=-1)
                aa=torch.sum(a**2,dim=-1)
                bb=torch.sum(b**2,dim=-1)
                cc=torch.sum(c**2,dim=-1)

                loss=F.relu(self.margin+aa+bb+cc-d1-d2-d3)
                return torch.mean(loss)


        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)
        
    def Nlet_string(self):
        return "abc"

    def explain(self):
        return "Ring loss with margin of "+str(self.margin)+". The formula is relu(margin+Norm(a)+Norm(b)+Norm(c)-D(a,b)-D(b,c)-D(c,a))."
