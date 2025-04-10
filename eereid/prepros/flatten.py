from eereid.prepros.prepro import prepro

try:
    import cv2
except ImportError:
    from eereid.importhelper import importhelper
    cv2=importhelper("cv2","resize","pip install opencv-python")
import numpy as np

class flatten(prepro):
    def __init__(self):
        super().__init__("flatten")

    def _apply_one(self, image):
        return np.reshape(image, -1)

    def _apply_special(self,eereid):
        eereid.input_shape=list(eereid.input_shape)
        eereid.input_shape[0]=np.prod(eereid.input_shape)
        eereid.input_shape=eereid.input_shape[:1]

    def save(self,pth,index):
        super().save(pth,index,size=size)

    def stage(self):return "general"
    def order(self):return 20

    def explain(self):
        return f"Flattening each input data into a vector"
