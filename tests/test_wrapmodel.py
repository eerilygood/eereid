import unittest
import keras
from eereid.models.wrapmodel import wrapmodel
from eereid.modifier.mods import mods

class SimpleModel(wrapmodel):
    def __init__(self):
        super().__init__("SimpleModel")
        
    def build_submodel(self, input_shape, mods):
        layers=mods("layer_count",3)
        activation=mods("activation","relu")
        filters=mods("filters",64)
        outputs=mods("output_size",100)
        kernelsize=mods("kernel_size",(3, 3))
        convcount=mods("conv_count",2)
        pool=mods("pool_size",2)

        inp=keras.layers.Input(shape=input_shape)
        q=inp
        for i in range(layers):
            for j in range(convcount):
                q=keras.layers.Conv2D(filters,kernelsize,activation=activation,padding="same")(q)
            if pool>1 and i<layers-1:
                q=keras.layers.MaxPool2D(pool_size=(pool,pool))(q)
        q=keras.layers.Flatten()(q)
        q=keras.layers.Dense(outputs,activation="linear")(q)

        self.submodel=keras.models.Model(inputs=inp,outputs=q)

    def explain(self):
        return "Creating a simple neural network model."
    
class WrapModelTest(unittest.TestCase):
    def test_build(self):
        model = SimpleModel()
        model.build((12,12,1),3,mods())