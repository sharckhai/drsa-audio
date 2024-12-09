from typing import Tuple, List

import torch
import torch.nn as nn

# TODO: change init to take path to config as input

class VGGType(torch.nn.Module):
    """Creates convolutional n eural netowork.
    
    Depending on the inputs to this class, feed forward CNNs of many architectures can be generated.
    This is especially a benefit when conduction grid search experiemnts."""
    
    def __init__(
        self, 
        n_filters: List[int] = [32, 64, 96, 128], 
        conv_kernel: Tuple[int,int] = (3,3),
        pool_kernels: List[Tuple[int,int]] = [(4,4), (2,4), (2,2), (2,2)], 
        n_dense: int = 512, 
        n_classes: int = 10,
        dropout: float = 0.2, 
        block_depth: int = 2,
        dense_depth: int = 2,
        input_size: Tuple[int,int] = (128,256),
        padding: str = 'same',
        stride: int = 1,
        conv_bn: bool = True, 
        dense_bn: bool = True,
    ) -> None:
        """Receive arguments and initialise the  NN.

        Args:
            n_filters (List[int]): number of filters per convolutional block.
            conv_kernel (Tuple[int,int]): Kernel size for convolutional layers.
            pool_kernels (List[Tuple[int,int]]): Tuple of Tuples with kernel sizes for maxpool layers.
            n_dense (int): Number of neurons in dense layers.
            n_classes (int): Number of classes in dataset.
            dropout (float): Dropout in dense layers (classification head).
            block_depth (int): Number of convolutional layers in each block.
            dense_depth (int): Number of dense layers.
            input_size (Tuple[int,int]): Input size to model.
            padding (str): Padding for conv filters.
            stride (int): Stride of convolutions.
            conv_bn (bool): Flag to insert batch norm in feature extractor.
            dense_bn (bool): Flag to insert batch norm in classification head.
        """
        super(VGGType, self).__init__()

        assert len(n_filters) == len(pool_kernels), 'number of filters for conv blocks and maxpool kernels have to be equal'
        
        # Conv NN, feature extractor
        blocks = []
        for i, filters in enumerate(n_filters):
            blocks.extend(
                get_conv_block_layers(
                    n_in=n_filters[i-1] if i>0 else 1, 
                    n_out=filters, 
                    kernel=conv_kernel, 
                    block_depth=block_depth, 
                    padding=padding, 
                    stride=stride, 
                    conv_bn=conv_bn
                )
            )
            blocks.extend([nn.MaxPool2d(pool_kernels[i])])

        # create feature extractor
        self.features = nn.Sequential(*blocks)

        # calculate input shape for classification head
        conv_out_shape = get_out_shape(
            input_size=input_size, 
            conv_kernel=conv_kernel, 
            pool_kernels=pool_kernels, 
            padding=padding, 
            out_filters=n_filters[-1], 
            block_depth=block_depth
        )
        # create classification head
        self.classifier = nn.Sequential(
            *get_dense_block_layers(
                n_in=conv_out_shape, 
                n_out=n_dense, 
                dropout=dropout, 
                depth=dense_depth, 
                dense_bn=dense_bn
            ),
            nn.Linear(n_dense, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # NOTE: hardcode flat features, the only purpose of num_flat_features is for grid search of best model architecture
        #x = x.view(-1, num_flat_features(x)) 
        x = x.view(-1, 2048)
        logits = self.classifier(x)
        return logits


def get_conv_block_layers(
    n_in: int, 
    n_out: int, 
    block_depth: int = 2, 
    kernel: Tuple[int,int] = (3,3), 
    stride: int = 1, 
    padding: int = 1, 
    padding_mode: str = 'zeros', 
    conv_bn: bool = True
) -> List[nn.Module]:
    """Creates list of layers for a convolutional block (= Conv2d -> BatchNorm2d -> ReLU)
    
    Args:
        n_in (int): input dimension
        n_out (int): out dimension (number of conv filters)
        block_depth (int): number of convolutional layers in each block
        kernel (tuple): kernel size of convolutional layers 
        padding (int): padding for conv filters
        stride (int): stride of convolution
    
    Returns:
        layers (List[nn.Module]): List of layers.
    """
    layers = []
    for i in range(block_depth):
        layers.extend(
            [nn.Conv2d(
                in_channels=n_in if i==0 else n_out, 
                out_channels=n_out, 
                kernel_size=kernel,
                stride=stride, 
                padding=padding, 
                padding_mode=padding_mode
            )] +
            ([nn.BatchNorm2d(num_features=n_out)] if conv_bn else []) +
            [nn.ReLU()]
        )
    return layers


def get_dense_block_layers(
    n_in: int, 
    n_out: int, 
    dropout: float, 
    depth: int = 2, 
    dense_bn: bool = True, 
) -> List[nn.Module]:
    """Creates list of layers for the classification head. 

    Each dense block is composed as Linear -> BatchNorm2d -> ReLU (-> Dropout).

    Args:
        n_in (int): Input dimension.
        n_out (int): Output dimension (number of conv filters).
        dropout (int): Dropout.
        depth (int): Number of dense layers in dense block.

    Returns:
        layers (List[nn.Module]): List of layers.
    """
    layers = []
    # create layers and add to list
    for i in range(depth):
        layers.extend(
            [nn.Linear(in_features=n_in if i==0 else n_out, out_features=n_out)] +
            ([nn.BatchNorm1d(n_out)] if dense_bn else []) +
            [nn.ReLU()]
        )
        # add dropout layer if wanted
        if dropout:
            layers.extend([nn.Dropout(dropout)])
    return layers
    

def get_out_shape(input_size: Tuple[int,int] = (128,216), 
                conv_kernel: Tuple[int,int] = (3,3), 
                pool_kernels: List[Tuple[int,int]] = [(4,4), (2,4), (2,2), (2,2)], 
                out_filters: int = 128,
                padding: int = 1, 
                stride: int = 1, 
                block_depth: int = 2
                ) -> int:
    """Calculates the output shape of the feature extractor.

    Args:
        input_size (tuple): Size of input image.
        conv_kernel (tuple): Kernel size for convolutional layers.
        pool_kernels (tuple): Tuple of Tuples with kernel sizes for maxpool layers.
        out_filters (int): Number of filters in last conolutional layer.
        block_depth (int): Number of convolutional layers in each block.
        padding (int): Padding for conv filters.
        stride (int): Stride of convolution.

    Returns:
        conv_out_shape (int): Shape of flattend output tensor of last convolutional layer
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


def num_flat_features(x: torch.Tensor):
    """Calculates flattend size of one input tensor contained in batch x.

    Args:
        x (torch.Tensor): Input tensor to dense layers of NN.

    Returns:
        num_features (int): Number of features of flattend input tensor to classifier.
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
