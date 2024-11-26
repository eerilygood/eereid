from eereid.models.wrapmodel import wrapmodel
import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, Conv2D, BatchNormalization, ReLU, Dropout, Layer, LeakyReLU, Lambda, MaxPooling2D

class DimReduceLayer(Layer):
    def __init__(self, in_channels, out_channels, nonlinear, **kwargs):
        super(DimReduceLayer, self).__init__(**kwargs)
        self.conv = Conv2D(
            out_channels, kernel_size=1, strides=1, padding='valid', use_bias=False
        )
        self.bn = BatchNormalization()
        
        if nonlinear == 'relu':
            self.activation = ReLU()
        elif nonlinear == 'leakyrelu':
            self.activation = LeakyReLU(alpha=0.1)
        else:
            self.activation = None

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x

class AdaptiveAveragePooling2D(Layer):
    def __init__(self, output_size, **kwargs):
        super(AdaptiveAveragePooling2D, self).__init__(**kwargs)
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def call(self, inputs):
        input_shape = inputs.shape
        h, w = input_shape[1], input_shape[2]
        pool_size = (math.ceil(h / self.output_size[0]), math.ceil(w / self.output_size[1]))
        return AveragePooling2D(pool_size=pool_size, strides=pool_size)(inputs)

class Bottleneck(Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = Conv2D(planes, kernel_size=1, strides=1, use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(
            planes, 
            kernel_size=3, 
            strides=stride, 
            padding='same', 
            use_bias=False
        )
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(
            planes * self.expansion, 
            kernel_size=1, 
            strides=1, 
            use_bias=False
        )
        self.bn3 = BatchNormalization()
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, inputs, training=False):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)

        return out

class PCB(wrapmodel):
    def __init__(self, loss, parts=6, reduced_dim=256, nonlinear='relu'):
        super().__init__("PCB")
        self.loss = loss
        self.parts = parts
        self.reduce_dim = reduced_dim
        self.non_linear = nonlinear
        self.inplanes = 64
        
    def explain(self):
        return f"PCB model implementation, using the base model ResNet."
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                Conv2D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False),
                BatchNormalization()
            ])

        layers_list = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes))

        return tf.keras.Sequential(layers_list)

    def build_submodel(self,input_shape, mods):
        training=mods("training", True)
        
        # Backbone network
        conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        bn1 = BatchNormalization()
        relu = ReLU()
        maxpool = MaxPooling2D(pool_size=3, strides=2, padding='same')
        
        layer1 = self._make_layer(Bottleneck, 64, 3)
        layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        layer4 = self._make_layer(Bottleneck, 512, 3, stride=1)
        
        # PCB layers
        parts_avgpool = AdaptiveAveragePooling2D((self.parts, 1))
        dropout = Dropout(0.5)
        conv5 = DimReduceLayer(512, self.reduce_dim, nonlinear=self.non_linear)
        
        ### Input flow
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Feature extraction
        x = conv1(inputs)
        x = bn1(x)
        x = relu(x)
        x = maxpool(x)
        x = layer1(x)
        x = layer2(x)
        x = layer3(x)
        x = layer4(x)
    
        # PCB-specific layers    
        v_g = parts_avgpool(x)

        if not training:
            v_g = tf.nn.l2_normalize(v_g, axis=1)
            return Lambda(lambda x: tf.reshape(x, [x.shape[0], -1]))(v_g)

        v_g = dropout(v_g)
        v_h = conv5(v_g)

        y = []
        for i in range(self.parts):
            v_h_i = Lambda(
                function=lambda x: tf.squeeze(x),
                output_shape=lambda input_shape: (input_shape[0], input_shape[2])
                )(v_h[:, i, :, :])
            y_i = Dense(self.reduce_dim, activation="softmax")(v_h_i)
            y.append(y_i)

        predictions = Lambda(function=lambda x: tf.concat(x, axis=-1))(y)

        # if self.loss == 'softmax':
        #     return y
        # elif self.loss == 'triplet':
        #     v_g = tf.nn.l2_normalize(v_g, axis=1)
        #     return y, tf.reshape(v_g, [v_g.shape[0], -1])
        # else:
        #     raise ValueError(f'Unsupported loss: {self.loss}')
        
        self.submodel = Model(inputs=inputs, outputs=predictions)
    
def pcb_p6(loss='softmax', **kwargs):
    model = PCB(
        loss=loss,
        parts=6,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    return model


def pcb_p4(loss='softmax', **kwargs):
    model = PCB(
        loss=loss,
        parts=4,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    return model