import torch
import torch.nn as nn

# Laplace normalization
class LplsNorm(nn.Module):
    def __init__(self):
        super(LplsNorm, self).__init__()

    def forward(self, adjacency_matrix):
        degree = torch.sum(adjacency_matrix, dim=-1)
        diag = torch.diag(torch.pow(degree, -0.5))
        out = diag.mm(adjacency_matrix).mm(diag)
        return out

# convolutional
class Convolution(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Convolution, self).__init__()
        self.LinearLayer = nn.Sequential(
            nn.Linear(in_dimension, out_dimension),
            nn.ReLU()
        )

    def forward(self, adjacency_matrix, feature):
        mf = feature + torch.mm(adjacency_matrix, feature)
        out = self.LinearLayer(mf)
        return out

class ConvolutionLayer(nn.Module):
    def __init__(self, in_dimension, embedding_dimension):
        super(ConvolutionLayer, self).__init__()
        self.LplsNorm = LplsNorm()

        self.Convolution = Convolution(in_dimension, embedding_dimension)

    def forward(self, adjacency_matrix, feature):
        adjacency_matrix = self.LplsNorm(adjacency_matrix)
        out = self.Convolution(adjacency_matrix, feature)
        return out

class GraphConvolution(nn.Module):
    # in_dimenssion: input_dimenssion
    def __init__(self, in_dimension, embedding_dimension):
        super(GraphConvolution, self).__init__()
        self.ConvolutionLayer = ConvolutionLayer(in_dimension, embedding_dimension)
        
        self.LinearLayer = nn.Sequential(
            nn.Linear(embedding_dimension, in_dimension),
            nn.ReLU(),
            nn.Linear(in_dimension, in_dimension)
        )

    def forward(self, matrix, feature):
        re = self.ConvolutionLayer(matrix, feature)
        rd = self.LinearLayer(re)
        
        return rd, re