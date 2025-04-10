from eereid.prepros.prepro import prepro
import numpy as np
import warnings

class rotations(prepro):
    def __init__(self,seed=42):
        super().__init__("rotations")
        self.seed=seed

    def apply(self, data, labels, eereid):
        datas=[]
        if data.shape[1]!=data.shape[2]:
            warnings.warn("The input data does not have equal width and height. Rotations may not work as expected.")

            datas.append(data)
            datas.append(np.rot90(data,2,axes=(1,2)))
        else:
            for i in range(4):
                datas.append(np.rot90(data,i,axes=(1,2)))
                # datas.append(np.rot180(data,i,axes=(2,3)))
                print(datas[-1].shape)
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
    def order(self):return 1

    def explain(self):
        return "Adds additional training images by rotating each training images. Quadruples the number of training images."
