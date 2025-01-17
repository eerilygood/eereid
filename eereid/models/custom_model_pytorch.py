from eereid.models.wrapmodel_pytorch import wrapmodel_pytorch
import numpy as np


class custom_model_pytorch(wrapmodel_pytorch):
    def __init__(self,model):
        super().__init__("custom_model")
        self.submodel=model

    def build_submodel(self,input_shape, mods):
        #assert (self.submodel.input_shape==input_shape), f"The provided model has a different input shape than the data ({self.submodel.input_shape} vs {input_shape})"
        pass

    def explain(self):
        return "Custom pytorch model wrapper gag"






