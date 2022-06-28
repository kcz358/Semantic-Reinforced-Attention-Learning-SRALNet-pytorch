# Semantic-Reinforced-Attention-Learning-SRALNet-pytorch
Pytorch implementation on SRALNet from [Semantic Reinforced Attention Learning for Visual Place Recognition](https://arxiv.org/abs/2108.08443)

The SRALNet, similar to the [NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/abs/1511.07247) also use a unified conv layer to process the extracted local features. Differently, the SRALNet does an extra intra cluster assignment to suppress irrelevant object.

The SRALNet.py in this file is a pytorch impelementation of the SRALNet. 
The implementation has refered to [NetVLAD-pytorch](https://github.com/lyakaap/NetVLAD-pytorch)

## Usage Example

'''
import torch
import torch.nn as nn
'''
