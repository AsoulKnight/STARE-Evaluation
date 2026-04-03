import numpy as np

weights = np.load("C:/1.lwBrown/Lee Lab/Evaluation_STARE/models/R50+ViT-B_16.npz")
print(len(weights.files))
print(weights.files[:20])