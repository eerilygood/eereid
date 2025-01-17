from eereid.losses.loss import loss

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class triplet_pytorch(loss):
    def __init__(self, margin=1.0):
        self.margin = margin
        super().__init__("triplet_pytorch")

    def build(self, mods):
        def func(y_true, y_pred):
            anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
            positive_dist = torch.sum((anchor - positive) ** 2, dim=-1)
            negative_dist = torch.sum((anchor - negative) ** 2, dim=-1)
            return torch.mean(F.relu(positive_dist - negative_dist + self.margin))

        return func

    def save(self, pth):
        super().save(pth, margin=self.margin)

    def Nlet_string(self):
        return "aab"

    def explain(self):
        return f"Triplet loss with margin of {self.margin}. The formula is relu(D(a,p)-D(a,n)+margin)."

