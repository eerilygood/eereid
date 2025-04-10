from eereid.prepros.prepro import prepro

try:
    import cv2
except ImportError:
    from eereid.importhelper import importhelper
    cv2=importhelper("cv2","resize","pip install opencv-python")
import numpy as np

class normalize(prepro):
    def __init__(self):
        super().__init__("normalize")

    def _apply_one(self, image):
        mx,mn=np.max(image),np.min(image)
        # im = ((image - mn) / (mx - mn)).astype(np.float32)
        return ((image-mn)/(mx-mn)).astype(np.float16) 

    def _apply_special(self,eereid):
        pass

    def save(self,pth,index):
        super().save(pth,index,topx=self.topx,topy=self.topy,botx=self.botx,boty=self.boty)

    def stage(self):return "general"
    def order(self):return 3

    def explain(self):
        return f"Normalizing input so that min=0,max=1"
