# STARE-Evaluation

This is an evaluation of UNet, TransUNet, and UNETR in STARE dataset.
It aims to find the best performing model on labelled retinal vessel image segmentation task using STARE dataset.
STARE dataset offers very limited amount of labelled retinal vessel images, so it needs a carefully preprocessing with data augmentation while not influencing its imaging structure.
Thus, I chose to do basic transformations and contrast change for augmentation.
The transunet and unetr don't perform well with slight change like changing normalization or activation function, they need bigger architechture change.
For pre-trained R50+ViT-B_16.npz, please download it from transunet github page.
