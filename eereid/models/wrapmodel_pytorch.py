from eereid.models.model import model
from eereid.losses import triplet_pytorch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary


class extending_layer(nn.Module):
    def forward(self, inputs):
        return inputs.unsqueeze(0)

class concat_layer(nn.Module):
    def forward(self, inputs):
        return torch.cat(inputs, dim=0)

class wrapmodel_pytorch(model):
    def __init__(self,name):
        super().__init__("wrap_"+name)

    def build_submodel(self, input_shape, mods):
        raise NotImplementedError

    # TODO this needs rethinking, model was not built as it should
    def build(self, input_shape, siamese_count, mods):
        self.build_submodel(input_shape, mods)

        # self.siamese_count = siamese_count
        self.input_shape = input_shape

        # self.extending_layer = extending_layer()
        # self.concat_layer = concat_layer()
        self.model = model_net(self.submodel, siamese_count)
    
    # TODO: try to make this work, custom loss and optimizer did not work as expected
    def compile(self, loss,optimizer, *args, **kwargs):
        # self.pytorch_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        loss_obj = triplet_pytorch(margin=1.0)
        self.pytorch_criterion = loss_obj.build(None)
        if optimizer == "adam":
            self.pytorch_optimizer = torch.optim.Adam(self.submodel.parameters(), lr=0.001)
        elif optimizer == "sgd":
            self.pytorch_optimizer = torch.optim.SGD(self.submodel.parameters(), lr=0.001, momentum=0.9)
        else:
            raise ValueError("optimizer not recognized")
        
        # self.pytorch_optimizer = optimizer
        self.model.set_optimizer(optimizer_wrapper(self.pytorch_optimizer))

    
    # some of the parameters in fit() have to be somehow passed
    def fit(self, triplets, labels, *args, **kwargs):
        print("fitting")
        print("triplets: ", triplets.shape)
        print("input_ shape:",self.input_shape)
        for epoch in range(2):  # loop over the dataset multiple times
            print("epoch: ", epoch)
            running_loss = 0.0
            print("triplets: ", triplets.shape)
            tripletsDataSet = triplet_dataset(triplets, labels)
            trainloader = torch.utils.data.DataLoader(tripletsDataSet, batch_size=32)
            print("trainloader: ",trainloader)
            for i, data in enumerate(trainloader, 0):
                
                # print("epoch: ", epoch)
                # print("i: ", i)
                # print('i: %d' % i, end='\r')
                inputs, labels = data

                self.pytorch_optimizer.zero_grad()

                # outputs = model_net(inputs) # might need to change this
                # print("inputs: ", inputs.shape)
                anchor, positive, negative = inputs[:, 0], inputs[:, 1], inputs[:, 2]

                anchor = anchor.permute(0, 3, 1, 2)
                positive = positive.permute(0, 3, 1, 2)
                negative = negative.permute(0, 3, 1, 2)
                anchor = anchor.float()
                positive = positive.float()
                negative = negative.float()
                # print("anchor: ", anchor.shape)
                # print("positive: ", positive.shape)
                # print("negative: ", negative.shape)
                
                outputs = torch.stack([self.submodel(anchor), self.submodel(positive), self.submodel(negative)], dim=0)
                outputs = outputs.requires_grad_(True)

                loss = self.pytorch_criterion(labels, outputs)
                loss.backward()
                self.pytorch_optimizer.step()

                running_loss += loss.item()
                # if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[epoch: %d, %5d/] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / 2000), end='\r')
                running_loss = 0.0

    def explain(self):
        return "Inheritable generic class creating siamese neural network wrapper"
    
    def summary(self,*args,**kwargs):
        print("submodel:")
        # summary(self.submodel,tuple([self.input_shape[0],self.input_shape[1],self.input_shape[2]]))
        print("model:")
        # summary(self.model,tuple([self.input_shape[0],self.input_shape[1],self.input_shape[2]]))


class model_net(nn.Module):
    def __init__(self,submodel,siamese_count):  
        super(model_net, self).__init__()
        self.submodel = submodel
        self.siamese_count = siamese_count

        # self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        
    def forward(self, x):
        samples = [x[:, i] for i in range(self.siamese_count)]
        samples = [self.submodel(sample) for sample in samples]
        samples = [extending_layer(sample) for sample in samples]
        outp = concat_layer(samples)
        return outp
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
class optimizer_wrapper(torch.optim.Optimizer):
    def __init__(self, pytorch_optimizer):
        super(optimizer_wrapper, self).__init__(pytorch_optimizer.param_groups, pytorch_optimizer.defaults)
        self.optimizer = pytorch_optimizer
        # self.optimizer = pytorch_optimizer
        # self.learning_rate = pytorch_optimizer.param_groups[0]['lr']
        self.learning_rate = learning_rate_class(pytorch_optimizer.param_groups[0]['lr'])
        
class learning_rate_class:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def assign(self, learning_rate):
        self.learning_rate = learning_rate
        
class triplet_dataset(torch.utils.data.Dataset):
    def __init__(self, triplets, labels):
        self.triplets = triplets
        self.labels = labels

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx], self.labels[idx]