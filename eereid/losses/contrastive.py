from eereid.losses.loss import loss

import numpy as np

from tensorflow.keras import backend as K

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    from eereid.importhelper import importhelper
    torch=importhelper("torch","wrapmodel_pytorch","pip install torch")

class contrastive(loss):
    def __init__(self, margin=1.0):
        self.margin = margin
        super().__init__("contrastive")

    def build(self,mods):

        def func(y_true, y_pred,use_pytorch=False):
            #print(y_true.shape,y_pred.shape)
            #exit()
            #aa: 1/2 D**2
            #ab: 1/2 max(0,margin-D)**2
            if not use_pytorch:
                a,b=y_pred[0],y_pred[1]
                dist=K.sum(K.square(a-b),axis=-1)
                return K.sum(y_true*dist+(1-y_true)*K.maximum(0.,self.margin-dist),axis=-1)
            else:
                a,b=y_pred[0],y_pred[1]
                dist=torch.sum((a-b)**2,dim=-1)
                return torch.sum(y_true*dist+(1-y_true)*F.relu(self.margin-dist),dim=-1)
            
        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)

    def Nlet_string(self):
        #usual contrastive would only use aa/ab, but this requires a y value. So for consistency we absorb both terms into one aab loss function
        return "aa/ab"

    def explain(self):
        return "Contrastive loss with margin of "+str(self.margin)+". The formula is 1/2 D**2 for same class pairs and 1/2 max(0,margin-D)**2 for different class pairs."
        

