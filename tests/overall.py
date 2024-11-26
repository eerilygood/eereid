from plt import *
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from eereid.eereid import eereid

from eereid.datasets.mnist import mnist
from eereid.datasets.fmnist import fmnist
from eereid.datasets.cifar import cifar

from eereid.distances.lN import lN
from eereid.distances.cosine_similarity import cosine_similarity
from eereid.distances.mahalanobis import mahalanobis
from eereid.distances.chisquare import chisquare
from eereid.distances.custom_distance import custom_distance

from eereid.losses.triplet import triplet
from eereid.losses.quadruplet import quadruplet
from eereid.losses.ringloss import ringloss
from eereid.losses.contrastive import contrastive
from eereid.losses.extended_triplet import extended_triplet
from eereid.losses.custom_loss import custom_loss

from eereid.models.simple_dense import simple_dense
from eereid.models.simple_conv import simple_conv
from eereid.models.custom_model import custom_model

from eereid.prepros.resize import resize
from eereid.prepros.blackwhite import blackwhite
from eereid.prepros.add_color import add_color
from eereid.prepros.rotations import rotations
from eereid.prepros.flips import flips   
from eereid.prepros.subsample import subsample


g=eereid(mnist(),lN(2),extended_triplet(),simple_conv())
g=g(subsample(0.1))
g["triplet_count"]=1000

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load the InceptionV3 model without the top layer (fully connected layer)
#base_model = InceptionV3(weights='imagenet', include_top=False)
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Add a global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected dense layer with, for example, 1024 units
x = Dense(1024, activation='relu')(x)

# Add a final dense layer for classification, for example with 10 classes
predictions = Dense(100, activation='linear')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

## Optionally, you can freeze the base_model layers to prevent them from being trained
#for layer in base_model.layers:
#    layer.trainable = False

# Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model to check the changes
model.summary()


g(custom_model(model))(resize((32,32)))(add_color())










#g=g(resize((16,16)))
#g=g(cifar())
#g=g(blackwhite())
#g=g(rotations())
#g=g(flips())
#g=g(fmnist())
#g=g(fmnist()+mnist())
#g=g(mahalanobis())
#g=g(chisquare())
#g=g(custom_distance(lambda x,y:np.max(np.abs(x-y))))
#g=g(quadruplet())
#g=g(ringloss())

#def loss(y_true, y_pred):
#    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
#    positive_dist = K.sum(K.square(anchor - positive), axis=-1)
#    negative_dist = K.sum(K.square(anchor - negative), axis=-1)
#    return K.sum(K.maximum(positive_dist - negative_dist,0), axis=-1)
#g(custom_loss(loss,"bbb"))

#g(contrastive())


acc=g.evaluate()
#acc=g.repeated_eval(3)
#acc=g.crossval_eval()
print(acc)

g.plot_embeddings()



plt.show()
