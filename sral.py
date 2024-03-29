import torch
import torch.nn as nn
import torch.nn.functional as F


class SRALNet(nn.Module):
    """SRALNet layer implementation"""

    def __init__(self, num_clusters=64, num_shadow = 4, dim=128, alpha=1500.0, centroids = None):
        """
        Args:
            num_clusters : int
                The number of clusters
            num_shadow : int
                The number of shadow centroids for each cluster
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            centroids : tensor
                Should be a tensor with shape (num_clusters, num_shadow + 1, dim)
                Each clusters first point should be your representative centroids and others are the shadow centroids
        """
        super(SRALNet, self).__init__()
        self.num_clusters = num_clusters
        self.num_shadow = num_shadow
        self.dim = dim
        self.alpha = alpha
        self.conv = nn.Conv2d(dim, self.num_clusters * (self.num_shadow + 1), kernel_size=(1, 1), bias=True)

        self.centroids = centroids
        self.rep_centroids = nn.Parameter(self.centroids[:, 0, :])
        self._init_params()

    def _init_params(self):
        """
        Remember to rewrite this method if you want to train this layer
        Initialize the centroids with your representative centroids and shadow centroids
        """
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids.view(self.num_clusters*(self.num_shadow + 1), -1)).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.view(self.num_clusters*(self.num_shadow + 1), -1).norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        
        # soft-assignment
        conv_out = self.conv(x).reshape(N, self.num_clusters, self.num_shadow + 1, -1)
        soft_alpha = F.softmax(conv_out[:,:,0:1,:], dim = 1)
        soft_beta = F.softmax(conv_out, dim = 2)[:,:,0:1,:]
        soft_assign = soft_alpha * soft_beta

        conv_flatten = x.view(N, C, -1)
        residual = conv_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                    self.rep_centroids.expand(conv_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        

        residual += soft_assign[:,:, 0, :].unsqueeze(2)*residual

        output = residual.sum(dim = -1)
        output = F.normalize(output, p = 2, dim = 2)
        output = output.view(x.size(0), -1)  # flatten
        output = F.normalize(output, p=2, dim=1)    

        return output, soft_assign
