from eereid.prepros.prepro import prepro
import numpy as np

class flips(prepro):
    def __init__(self,seed=42):
        super().__init__("flips")
        self.seed=seed

    def apply(self, data, labels, eereid):
        datas=[]
        datas.append(data)
        datas.append(np.flip(data,1))
        datas.append(np.flip(data,2))
        datas.append(np.flip(datas[-1],1))
        datas=np.concatenate(datas,axis=0)
        labels=np.tile(labels,4)

        rnd=np.random.RandomState(self.seed)
        idx=rnd.permutation(datas.shape[0])
        datas=datas[idx]
        labels=labels[idx]

        return datas,labels

    def save(self,pth,index):
        super().save(pth,index,seed=self.seed)

    def stage(self):return "train"
    def order(self):return 2

    def explain(self):
        return "Adds additional training images by flipping each training images. Quadruples the number of training images."
