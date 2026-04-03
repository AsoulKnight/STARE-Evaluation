import torch
import torch.nn as nn
import numpy as np

from models.vit_seg_modeling import VisionTransformer, CONFIGS


class TransUNet(nn.Module):
    def __init__(self, img_size=512, pretrained_path=None):
        super().__init__()

        config = CONFIGS["R50-ViT-B_16"]
        ## a minimal change here
        config.n_skip = 0
        #####
        config.n_classes = 1
        config.activation = "sigmoid"

        self.model = VisionTransformer(
            config,
            img_size=img_size,
            num_classes=1
        )

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        print(f"Loading pretrained weights from: {path}")
        weights = np.load(path)
        self.model.load_from(weights)
        print("✅ Pretrained weights loaded")

    def forward(self, x):
        return self.model(x)