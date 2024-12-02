import random
from typing import Tuple
import warnings
warnings.filterwarnings("ignore", message="MPS: no support for int64 repeats mask, casting it to int32")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from zennit.composites import Composite
from zennit.attribution import Gradient

from cxai.xai.explain.attribute import lrp_output_modifier
from cxai.utils.dataloading import get_songlist, get_toy_samplelist, shuffle_and_truncate_databatch, Loader


def preprocess_data(
    model: nn.Sequential,
    input_batch: np.ndarray | torch.Tensor,
    composite: Composite, 
    layer_idx: int,
    class_idx: int,
    num_locations: int | None = None,
    one_hot_encoded: bool = False,
    device: str | torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates the training data for DRSA optimization algorithm.

    Extracts activation and relevance maps from a CNN and prepares the dataset through DRSA optimization.
    Activation and relevance vectors are extracted with hooks at the specified layer.

    Args:
        model (nn.Sequential): CNN model.
        input_batch (np.ndarray | torch.Tensor): Input instances of one class for the CNN (spectrograms).
        composite (zennit.composites.Composite): A rule composite defining LRP rules for each model layer
                                                 to backpropagate relevances.
        layer_idx (int): Layer at which to extract activation and context vectors.
        class_idx (int): Class of the insatnces in the input batch (for which subspaces should be trained).
        num_locations (int | None, optional): Number of vectors to sample from each set of feature maps.
        one_hot_encoded (bool, optional): If True, the total output relevance of the specified class to 
                                          perform LRP is set to 1. If False, the true output logit is set 
                                          as total relevance.
        device (str | torch.device, optional): Device.

    Returns:
        tuple: A tuple containing:
            - activation_maps (torch.Tensor): Activation maps. 
            - relevance_maps (torch.Tensor): Relevance maps.
    """
    if isinstance(device, str): device = torch.device(device)

    # input_batch.shape: [batch, channel, height, width]
    if isinstance(input_batch, np.ndarray):
        input_batch = torch.tensor(input_batch).to(device)
    else:
        input_batch.to(device)
    batch_size = input_batch.size(0)
    # extract layer to register hook
    layer = model.features[layer_idx]
    # extract activation and relevance maps at specified layer
    # _maps.shape: [batch, d, filter_size, filter_size]
    activation_maps, relevance_maps = get_intermediate(
        model, 
        layer, 
        input_batch, 
        composite,
        class_idx, 
        one_hot_encoded=one_hot_encoded
    )

    print('Extrcating activation and context vectors at several spatial locations...')

    if num_locations:  # CASE: used for model training
        # sample num_locations random locations for each set of feature maps in the batch
        idcs_batch = sample_spatial_locations(batch_size, activation_maps.size()[-2:], num_locations)
        # extract vectors from maps
        activation_vectors = get_vectors_from_maps(activation_maps, idcs_batch)
        relevance_vectors = get_vectors_from_maps(relevance_maps, idcs_batch)
    else: # CASE: used for inference. We use all vectors contained in a set of feature maps
        # _vectors.shape: [batch, filter_size*filter_size, d]
        b, N = activation_maps.size(0)[:2]
        activation_vectors = activation_maps.reshape(b, N, -1).transpose(-2,-1)
        relevance_vectors = relevance_maps.reshape(b, N, -1).transpose(-2,-1)

    # compute context vectors from activations and relevances
    context_vectors = compute_context_vectors(activation_vectors, relevance_vectors)
    print('-'*10, '\nData preprocessing done!')
    return activation_vectors, context_vectors


def store_hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
    """Hook to extract activation maps and relevance maps from a model layer.
    
    Args:
        module (nn.Module): module (layer) to register the hook
        input (torch.Tensor): input to the module
        output (torch.Tensor): output of the module (i.e., activations)
    """
    # keep the output tensor gradient, even if it is not a leaf-tensor
    module.output = output
    # retain grad to save relevances
    output.retain_grad()


def get_intermediate(
    model: nn.Sequential, 
    input_batch: torch.Tensor, 
    composite: Composite, 
    layer: nn.Module,
    class_idx: int, 
    attr_batch_size: int = 64, 
    one_hot_encoded: bool = False
) -> Tuple[torch.Tensor,torch.Tensor]:
    """Extarct activation and relevance maps at given layer form a model.
    
    Propagates dtabatch through the network and backpropagates gradients up to the inputs. 
    Extracts activation maps and relevance maps at defined layer with the  use of a store hook.
    
    Args:
        model (nn.Sequential): CNN model.
        input_batch (np.ndarray | torch.Tensor): Input instances of one class for the CNN (spectrograms).
        composite (zennit.composites.Composite): A rule composite defining LRP rules for each model layer
                                                 to backpropagate relevances.
        layer (nn.Module): nn.Module to extract the activation and relevance maps from.
        attr_batch_size (int, optional): Size of minibatches that get attributed through the network.
        one_hot_encoded (bool, optional): If True, the total output relevance of the specified class to 
                                          perform LRP is set to 1. If False, the true output logit is set 
                                          as total relevance.
    Returns:
        tuple: A tuple containing:
            - activation_maps (torch.Tensor): Activation maps. 
            - relevance_maps (torch.Tensor): Relevance maps.
    """
    # get total batch size of inputs
    batch_size = input_batch.size(0)
    
    # process data in smaller batches to avoid overloading the gpu by storing intermediate outputs
    num_mini_batches = (batch_size + attr_batch_size - 1) // attr_batch_size

    activation_maps, relevance_maps = [], []
    # register lrp hooks
    with Gradient(model, composite) as attributor:

        # register own store hook to save intermediate representations during forward pass 
        # (retain grad saves grad in backward pass)
        handles = [layer.register_forward_hook(store_hook)]
        with tqdm(total=batch_size, desc='Extracting activation and relevance maps') as pbar:
            for i in range(num_mini_batches):

                # seperate input batch in smaller batches to avoid gpu overhead
                batch = input_batch[i*attr_batch_size:min((i+1)*attr_batch_size, input_batch.size(0))]
                batch = batch.requires_grad_(True)

                # compute the relevances
                _, _ = attributor(
                    batch, 
                    lrp_output_modifier(
                        class_idx, 
                        one_hot_encoded=one_hot_encoded
                    )
                )
                activation_maps.append(layer.output.detach().to(device=batch.device))
                relevance_maps.append(layer.output.grad.detach().to(device=batch.device))
                # reset the store hook variable to None
                layer.output = None
                pbar.update(batch.shape[0])

    # remove the store-hook
    for handle in handles:
        handle.remove()

    # extract activation and relevance maps 
    activation_maps = torch.concat(activation_maps, dim=0).requires_grad_(False)
    relevance_maps = torch.concat(relevance_maps, dim=0).requires_grad_(False)
    return activation_maps, relevance_maps


def compute_context_vectors(
    activation_vectors: torch.Tensor, 
    relevance_vectors: torch.Tensor, 
) -> torch.Tensor:
    """Compute context vectors (model response to activations from the view of relevances).
    
    Args:
        activation_vectors (torch.Tensor): Activation vectors.
        relevance_vectors (torch.Tensor): Context vectors.

    Returns:
        context_vectors (torch.Tensor): Context vectors.
    """
    # add epsilon to avoid division by zero
    return relevance_vectors / (activation_vectors + 1e-7)


def sample_spatial_locations(
    batch_size: int, 
    map_size: Tuple[int,int], 
    num_locations: int
) -> np.ndarray:
    """Samples random locations for each instance in the batch.
    
    Args:
        batch_size (int): Batch size.
        map_size (Tuple[int,int]): Size of feature maps (height,width).
        num_locations (int): Number of locations to sample from all feature map loactions.
                             Total number equals height*width of the set of feature maps.
    
    Returns:
        idcs_batch (np.ndarray): Locations to sample vectors from the set of feature maps.
    """
    idcs_batch = np.zeros((batch_size, num_locations), dtype=int)
    # idcs for flattened feature maps
    for i in range(batch_size):
        idcs_batch[i, :] = np.random.choice(map_size[0]*map_size[1], num_locations, replace=False)
    return idcs_batch
    

def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Normalizes activation and context vectors to enable more 
    stable training of subspaces. see DRSA paper.
    
    Args:
        vectors (torch.Tensor): Batch of vectors to normalize.

    Returns:
        vectors_normalized (torch.Tensor): Normalized vectors (normaloization method of DRSA paper).
    """
    d = vectors.size()[-1]
    E = torch.sqrt(torch.mean(torch.square(vectors)))
    return vectors / E / d**0.25


def get_vectors_from_maps(maps: torch.Tensor, idcs_batch: np.ndarray) -> torch.Tensor:
    """Extraxts vectors at specified lcoations from the set of feature maps.

    Suppose a set of feature maps with the shape (height x width x num_filters). We sample
    vectors with shape (num_filters,) form the set of feature maps. The total number of 
    vectors to sample from such a set of feature maps is therefore height*width.
    
    Args:
        maps (torch.Tensor): A abtch with sets of feature maps. Shape: [batch, d, width, height].
        idcs_batch (np.ndarray): Locations to sample vectors from. Shape: [batch, num_locations].

    Returns:
        vectors (torch.Tensor): Feature vectors. Shape: [batch*num_locations, d].
    """
    # [batch, d, height, width]
    batch_size, d, _, _ = maps.size()
    # [batch, d, height * width]
    maps = maps.reshape(batch_size, d, -1)
    # [batch, d, num_locs]
    vectors = maps[np.arange(batch_size)[:,None], :, idcs_batch]
    # [batch * num_locs, d]
    vectors = vectors.transpose(-2, -1).reshape(-1, d)
    return vectors


# BKUP code (unused)


def grid_locs(
    batch_size: int, 
    map_size: Tuple[int,int], 
    num_locations: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Supposes squared feature maps. Samples evelny spaces 
    locations (grid) across the feature map.
    
    Args:
        batch_size (int): batch size
        map_size (Tuple[int,int]): size of the feature maps (height,width)
        num_locations (int): number of locations to sample from feature map loactions

    Returns:
        Tuple[np.ndarray, np.ndarray]: grid lcoations to sample vectors from feature maps
    """

    assert num_locations%(num_locations**.5) == 0, 'num_locs has to be a perfect square'

    # get num_samples per axis
    filter_size = map_size[0]
    k = int(num_locations**.5)
    # get evenly spaced lcoations
    even_locs = get_evenly_spaced_locs(filter_size, k)
    
    # repeat sample points to create k**2 grid points
    grid_locs_x, grid_locs_y = np.meshgrid(even_locs, even_locs)
    # repeat arrays batch_size times
    x_grid_locs_batch = np.tile(grid_locs_x.flatten(), (batch_size, 1))
    y_grid_locs_batch = np.tile(grid_locs_y.flatten(), (batch_size, 1))
    return x_grid_locs_batch, y_grid_locs_batch


def get_evenly_spaced_locs(filter_size: int, k: int) -> np.ndarray:
    """Samples evenly spaced nubers from 0-filter_size."""

    # create array of sample points
    step_size = int(filter_size/k)
    grid_locs = np.arange(k)*step_size
    grid_locs = grid_locs + int(step_size/2)
    return grid_locs


def get_vectors_from_locs(maps, x_locs, y_locs) -> torch.Tensor:
    """Extracts vectors from feature maps at given x and y location."""

    # x_locs, y_locs shape: [batch_size, num_locs]
    batch_size, d, _, _ = maps.size()
    # [batch, d, num_locs]
    vectors = maps[np.arange(batch_size)[:,None], ..., x_locs, y_locs]
    # [batch * num_locs, d]
    vectors = vectors.transpose(-2, -1).reshape(-1, d)
    return vectors


# TODO: move these functions to another file

def get_songs_toy(datapath, sample_class, split=None, N=None):
    """Loading all samples of specific genre. Num chunks = 10. 
    Then we shuffle the data and truncate the batch if defined."""

    paths_to_songs = get_toy_samplelist(datapath, sample_class, split)
    if N is not None:
        random.shuffle(paths_to_songs)
        paths_to_songs = paths_to_songs[:N]

    loader = Loader(case='toy')
    data_batch = []
    songs = []

    # load samples as mel spectrograms
    for path_to_song in tqdm(paths_to_songs, total=len(paths_to_songs), desc="Loading samples from disk"):

        mels = loader.load(path_to_song) 
        data_batch.extend(mels.detach())
        songs.extend(path_to_song)
        
    data_batch_tensor = torch.stack(data_batch, dim=0)
    return data_batch_tensor, songs


# TODO: combine with get_songs_new_last from xai.prep

def get_songs_drsa(datapath, sample_class, excluded_folds=None, N=None, num_folds=5):
    """Loading all samples of specific genre. Num chunks = 10. Then 
    we shuffle the data and truncate the batch if defined."""

    paths_to_songs = get_songlist(datapath, sample_class, excluded_folds, num_folds=num_folds)

    loader = Loader()
    num_chunks = 10
    data_batch = []
    songs = []

    # load samples as mel spectrograms
    for path_to_song in tqdm(paths_to_songs, total=len(paths_to_songs), desc="Loading samples from disk"):

        # if path_to_song.endswith('hiphop.00036.wav') or path_to_song.endswith('hiphop.00038.wav'): continue 
        mels = loader.load(path_to_song, num_chunks=num_chunks) 
        data_batch.extend(mels)
        songs.extend([path_to_song for _ in range(num_chunks)])

    # create tensor from list
    data_batch_tensor = torch.stack(data_batch, dim=0)
    # shuffle bacth randomly
    if N:
        data_batch_tensor, songs = shuffle_and_truncate_databatch(data_batch_tensor, songs, N)
    
    return data_batch_tensor, songs
