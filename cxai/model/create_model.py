import numpy as np
import torch
import torch.nn as nn


class VGGType(torch.nn.Module):
    
    def __init__(self, 
                n_filters: tuple = (32, 64, 96, 128), 
                conv_kernel: tuple = (3,3),
                pool_kernels: tuple = ((4,4), (2,4), (2,2), (2,2)), 
                n_dense: int = 512, 
                n_classes: int = 10,
                dropout: float = 0.2, 
                block_depth: int = 2,
                dense_depth: int = 2,
                input_size: tuple = (128,256),
                padding: str = 'same',
                stride: int = 1,
                conv_bn=True, 
                dense_bn=True,
                **kwargs
                ):
        r"""
        Receive arguments and initialise the  NN.

        Parameters:
        ----------
        n_filters: Tuple[int, int, int, int]
            number of filters per convolutional block
        conv_kernel: Tuple[int, int]
            Kernel size for convolutional layers
        pool_kernels: Tuple[Tuple[int, int], Tuple[int, int], ...]
            Tuple of Tuples with kernel sizes for maxpool layers
        n_dense: int
            number of neurons in dense layers
        dropout: float
            dropout in dense layers (classification head)       
        block_depth: int
            number of convolutional layers in each block
        padding: int or string
            padding for conv filters
        stride: int
            stride of convolution
        """
        
        super(VGGType, self).__init__()

        assert len(n_filters) == len(pool_kernels), 'number of filters for conv blocks and maxpool kernels have to be equal'
        
        # Conv NN, feature extractor
        blocks = []
        for i, filters in enumerate(n_filters):
            blocks.extend(get_conv_block_layers(n_in=n_filters[i-1] if i>0 else 1, n_out=filters, kernel=conv_kernel, 
                                                block_depth=block_depth, padding=padding, stride=stride, conv_bn=conv_bn))
            blocks.extend([nn.MaxPool2d(pool_kernels[i])])

        # create feature extractor
        self.features = nn.Sequential(*blocks)

        # calculate input shape for classification head
        conv_out_shape = get_out_shape(input_size=input_size, conv_kernel=conv_kernel, pool_kernels=pool_kernels, 
                                       padding=padding, out_filters=n_filters[-1], block_depth=block_depth)

        # create classification head
        self.classifier = nn.Sequential(
            *get_dense_block_layers(n_in=conv_out_shape, n_out=n_dense, dropout=dropout, depth=dense_depth, dense_bn=dense_bn),
            nn.Linear(n_dense, n_classes)
        )


    def forward(self, x):

        # extract features
        x = self.features(x)

        # TODO: hardcode flat features

        # reshape activation map to flat input for dense layers
        x = x.view(-1, num_flat_features(x))   ################################# HARDCODE THIS !!!!!!!!!!!!!!!!!!!
        #print(num_flat_features(x))
        #x = x.view(-1, 2048)

        # feed forward
        logits = self.classifier(x)

        return logits



def get_conv_block_layers(n_in, n_out, block_depth=2, kernel: tuple = (3,3), stride: int = 1, padding: int = 1, padding_mode='zeros', conv_bn=True):
    r"""
    Creates list of layers for a convolutional block. 
    Conv2d -> BatchNorm2d -> ReLU

    Parameters:
    ----------
    n_in: int
        input dimension
    n_out: int
        out dimension (number of conv filters)
    block_depth: int
        number of convolutional layers in each block
    kernel: Tuple[int, int]
        kernel size of convolutional layers 
    padding: int or string
        padding for conv filters
    stride: int
        stride of convolution
    """
    layers = []
    for i in range(block_depth):
        layers.extend(
            [nn.Conv2d(in_channels=n_in if i==0 else n_out, out_channels=n_out, kernel_size=kernel, stride=stride, padding=padding, padding_mode=padding_mode)] +
            ([nn.BatchNorm2d(num_features=n_out)] if conv_bn else []) +
            [nn.ReLU()]
        )
    return layers


def get_dense_block_layers(n_in, n_out, dropout, depth=2, dense_bn=True, **kwargs):
    r"""
    Creates list of layers for the classification head. 
    Linear -> BatchNorm2d -> ReLU (-> Dropout)

    Parameters:
    ----------
    n_in: int
        input dimension
    n_out: int
        out dimension (number of conv filters)
    dropout: int
        dropout
    depth: int
        number of dense layers in dense block
    """
    layers = []
    # create layers and add to layers
    for i in range(depth):
        layers.extend(
            [nn.Linear(in_features=n_in if i==0 else n_out, out_features=n_out)] +
            ([nn.BatchNorm1d(n_out, **kwargs)] if dense_bn else []) +
            [nn.ReLU()]
        )
        # add dropout layer if wanted
        if dropout:
            layers.extend([nn.Dropout(dropout)])

    return layers
    


def get_out_shape(input_size=(128,216), 
                conv_kernel=(3,3), 
                pool_kernels=((4,4), (2,4), (2,2), (2,2)), 
                out_filters: int = 128,
                padding=1, 
                stride=1, 
                block_depth=2):
    r"""
    Calculates the output shape of the feature extractor.

    Parameters:
    ----------
    input_size: Tuple[int, int]
        Size of input image
    conv_kernel: Tuple[int, int]
        Kernel size for convolutional layers
    pool_kernels: Tuple[Tuple[int, int], Tuple[int, int], ...]
        Tuple of Tuples with kernel sizes for maxpool layers
    out_filters: int
        Number of filters in last conolutional layer
    padding: int or string
        padding for conv filters
    stride: int
        stride of convolution
    block_depth: int
        number of convolutional layers in each block

    Returns:
    -------
    conv_out_shape: int
        Shape of flattend output tensor of last convolutional layer
    """

    # transform padding to integer if str
    padding = 1 if padding=='same' else 0

    tensor_size = input_size

    for pool_kernel in pool_kernels:
        for _ in range(block_depth):
            # conv
            outx = ((tensor_size[0] - conv_kernel[0] + 2*padding) / stride) + 1
            outy = ((tensor_size[1] - conv_kernel[1] + 2*padding) / stride) + 1
            tensor_size = (outx, outy)
        # pooling
        outx = int((outx - (pool_kernel[0] - 1) - 1) / pool_kernel[0] + 1)
        outy = int((outy - (pool_kernel[1] - 1) - 1) / pool_kernel[1] + 1)
        tensor_size = (outx, outy)

    return tensor_size[0] * tensor_size[1] * out_filters



def num_flat_features(x):
    r"""
    Calculates flattend size of input tensor x for reshaping into flat tensor

    Parameters:
    ----------
    x: torch.Tensor
        Input tensor to dense layers of NN

    Returns:
    -------
    num_features: int
        number of features of flattend input tensor to classifier
    """
    size = x.size()[1:]  # all dimensions except the batch dimension

    num_features = 1
    for s in size:
        num_features *= s
        
    return num_features




"""class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out, block_depth=2, kernel: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1, padding_mode='zeros'):
        super().__init__()
        
        layers = []
        for i in range(block_depth):
            layers.extend([
                nn.Conv2d(in_channels=n_in if i==0 else n_out, out_channels=n_out, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode),
                nn.BatchNorm2d(num_features=n_out),
                nn.ReLU(),
            ])

        # create sequential model of this block
        self.block = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.block(x)


class DenseBlock(nn.Module):
    def __init__(self, n_in, n_out, dropout, depth=2):
        super().__init__()

        layers = []
        # create layers and add to layers
        for i in range(depth):
            layers.extend([
                nn.Linear(in_features=n_in if i==0 else n_out, out_features=n_out),
                nn.BatchNorm1d(n_out),
                nn.ReLU(),
            ])
            # add dropout layer if wanted
            if dropout:
                layers.extend([nn.Dropout(dropout)])

        # create sequential model of this block
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)"""


