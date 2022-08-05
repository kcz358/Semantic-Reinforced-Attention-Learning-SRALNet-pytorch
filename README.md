# Semantic-Reinforced-Attention-Learning-SRALNet-pytorch

Project Not finished

## Current Problem
- [ ] issues for setting weights for the convolutional layer. Might have some difference compare to the article


## Introduction
Pytorch implementation on SRALNet from [Semantic Reinforced Attention Learning for Visual Place Recognition](https://arxiv.org/abs/2108.08443)

The SRALNet, similar to the [NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/abs/1511.07247), also use a unified conv layer to process the extracted local features. Differently, the SRALNet does an extra intra cluster assignment to suppress irrelevant object.

- The SRALNet.py in this file is a pytorch impelementation of the SRALNet. <br>
- The implementation has refered to [NetVLAD-pytorch](https://github.com/lyakaap/NetVLAD-pytorch), [NetVLAD-training](https://github.com/Nanne/pytorch-NetVlad) for the basic structure of the NetVLAD <br>
- Pretrained DeepLabV3 model on Cityscapes can be found in site [deeplabv3](https://github.com/fregu856/deeplabv3)

## Usage Example


```python
import torch
import torch.nn as nn
from SRALNet import SRALNet

#vgg16 backbone
encoder = models.vgg16(pretrained=True)

# capture only feature part and remove last relu and maxpool
layers = list(encoder.features.children())[:-2]

for l in layers[:-5]: 
    for p in l.parameters():
        p.requires_grad = False

encoder = nn.Sequential(*layers)

#Suppose we have 40 pictures as input
input_image = torch.rand((40,3,224,224))

#transform the image for the vgg16
pre_process = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
input_image = pre_process(input_image)

#Use CNN backbone to extract local features first
image_encode = encoder(input_image)

sral = SRALNet(dim = 512)
sral(image_encode)
#output shape (40, 64*512)
```

