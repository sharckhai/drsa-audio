import os
import sys
import random
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import register_module_forward_hook
from torch.utils.data import DataLoader, TensorDataset
import pickle
from torch import mps
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="MPS: no support for int64 repeats mask, casting it to int32")

import zennit
from zennit.attribution import Gradient

from cxai.xai.explain.attribute import lrp_output_modifier
from cxai.utils.all import get_songlist, load_sample, get_toy_samplelist, shuffle_and_truncate_databatch, Loader


def preprocess_data(model: nn.Sequential,
                    input_batch: torch.Tensor,
                    composite: zennit.composites, 
                    layer_idx: int,
                    class_idx: int,
                    num_locations: int = None,
                    epsilon: float = 1e-7,
                    case = 'gtzan',
                    device: torch.device = torch.device('cpu'),
                    scaled_output: bool = False,
                    static = False,
                    ) -> tuple:
    """
    Returns:
    --------
    list of torch.Tensors
        activtion vectors and con text vectors
    """

    # [batch, channel, height, width]
    input_batch = torch.tensor(input_batch).to(device) if isinstance(input_batch, np.ndarray) else input_batch.to(device)
    batch_size = input_batch.size(0)
    # extract layer to register hook
    layer = model.features[layer_idx]

    # [batch, d, filter_size, filter_size]
    activation_maps, relevance_maps = get_intermediate(model, layer, input_batch, composite, class_idx, scaled_output=scaled_output)

    #batch_size, d, filter_size, filter_size = activation_maps.size()

    print('Extrcating activation and context vectors at several spatial locations...')
    if num_locations:
        """if static:
            # sample num_locations random spatial location to extract vectors from
            idcs_batch = sample_spatial_location_old(activation_maps.size()[-2:], num_locations)
            print(idcs_batch.shape)
            activation_vectors = get_vectors_from_maps_old(activation_maps, idcs_batch)
            relevance_vectors = get_vectors_from_maps_old(relevance_maps, idcs_batch)
        else:"""
        #idcs_batch = sample_spatial_location(batch_size, activation_maps.size()[-2:], num_locations)
        #print(idcs_batch.shape)
        #activation_vectors = get_vectors_from_maps(activation_maps, idcs_batch)
        #relevance_vectors = get_vectors_from_maps(relevance_maps, idcs_batch)

        """
        Shapes
        GTZAN: (128, 64, 32, 16, 8)
        TOY:   (64,  32, 16,  8, 4)
        """

        if static==0:

            x_locs, y_locs = grid_locs(batch_size, activation_maps.size()[-2:], num_locations)
            activation_vectors = get_vectors_from_maps_halfgrid(activation_maps, x_locs, y_locs)
            relevance_vectors = get_vectors_from_maps_halfgrid(relevance_maps, x_locs, y_locs)

        elif static==1:
            # sample time grid, random freq locs
            x_locs, y_locs = sample_spatial_locations_halfgrid(batch_size, activation_maps.size()[-2:], num_locations, case='time')
            activation_vectors = get_vectors_from_maps_halfgrid(activation_maps, x_locs, y_locs)
            relevance_vectors = get_vectors_from_maps_halfgrid(relevance_maps, x_locs, y_locs)

        elif static==2:
            # sample freq grid, random time locs
            x_locs, y_locs = sample_spatial_locations_halfgrid(batch_size, activation_maps.size()[-2:], num_locations, case='freq')
            activation_vectors = get_vectors_from_maps_halfgrid(activation_maps, x_locs, y_locs)
            relevance_vectors = get_vectors_from_maps_halfgrid(relevance_maps, x_locs, y_locs)
        
        else:
            # sample all random locations
            idcs_batch = sample_spatial_location(batch_size, activation_maps.size()[-2:], num_locations)
            activation_vectors = get_vectors_from_maps(activation_maps, idcs_batch)
            relevance_vectors = get_vectors_from_maps(relevance_maps, idcs_batch)


        #activation_maps = activation_maps.view(batch_size, d, -1)
        #relevance_maps = relevance_maps.view(batch_size, d, -1)
        #context_maps = compute_context_vectors(activation_maps, relevance_maps, epsilon)

        #idcs = get_best_locs(context_maps, num_locations=num_locations)
        #activation_vectors = get_vectors_from_maps_old(activation_maps, idcs)
        #context_vectors = get_vectors_from_maps_old(context_maps, idcs)

    else:
        # case to get prototypes, we dont sample random locations but instead use all data (all locations). This equals just reshaping the maps into vectors.
        # [batch, filter_size*filter_size, d]
        activation_vectors = activation_maps.reshape(activation_maps.size(0), activation_maps.size(1), -1).transpose(-2,-1)
        relevance_vectors = relevance_maps.reshape(activation_maps.size(0), activation_maps.size(1), -1).transpose(-2,-1)
        # compute context vectors
        #context_vectors = compute_context_vectors(activation_vectors, relevance_vectors, epsilon)

    context_vectors = compute_context_vectors(activation_vectors, relevance_vectors, epsilon)

    print('-'*10)
    print('Data preprocessing done!')

    return (activation_vectors, context_vectors)


def grid_sampler(n, channels, num_locations):
    """
    Suppose our feature maps are squared in size.
    """

    # how many grids per axis
    k = int(num_locations**.5)
    # size of sampling kernels
    m = n // k

    # create kernel
    kernel = torch.zeros((1, 1, m, m), dtype=torch.float)  # [out_channels, in_channels, height, width]
    # set middle value to 1
    kernel[0, :, m // 2, m // 2] = 1
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel, m

def get_vectors_from_maps_grid(maps, kernel, stride) -> torch.Tensor:
    
    # [batch, d, height, width]
    batch_size, d, _, _ = maps.size()

    # sample middle point for each smaller grid, [batch, d, num_locs]
    vecs = F.conv2d(maps, kernel.to(maps), stride=stride, groups=d).reshape(batch_size, d, -1)

    # [batch * num_locs, d]
    vecs = vecs.transpose(-2, -1).reshape(-1, d)
    return vecs

def grid_locs(batch_size, map_size, num_locations):

    assert num_locations%(num_locations**.5) == 0, 'num_locs has to be a perfect square'

    # get num_samples per axis
    filter_size = map_size[0]
    k = int(num_locations**.5)

    # get evenly spaced lcoations
    even_locs = get_evenly_spaced_locs(filter_size, k)

    print(even_locs)
    
    # repeat sample points to create k**2 grid points
    grid_locs_x, grid_locs_y = np.meshgrid(even_locs, even_locs)
    # repeat arrays batch_size times
    x_grid_locs_batch = np.tile(grid_locs_x.flatten(), (batch_size, 1))
    y_grid_locs_batch = np.tile(grid_locs_y.flatten(), (batch_size, 1))

    return x_grid_locs_batch, y_grid_locs_batch


def get_evenly_spaced_locs(filter_size, k):

    # create array of sample points
    step_size = int(filter_size/k)
    grid_locs = np.arange(k)*step_size
    grid_locs = grid_locs + int(step_size/2)

    return grid_locs



def sample_spatial_locations_halfgrid(batch_size, map_size, num_locations, case='time') -> np.ndarray:

    """
    For now we assume squared feature maps. 
    """

    assert num_locations%(num_locations**.5) == 0, 'num_locs has to be a perfect square'

    if case=='time':
        filter_size_static = map_size[1] # sample on a grid from time axis
        filter_size_rand = map_size[0] # sample random on freq axis
    else:
        filter_size_static = map_size[0] # sample on a grid from frequency axis
        filter_size_rand = map_size[1] # sample random on time axis

    # get num_samples per axis
    samples_per_axis = int(num_locations**.5)

    # create array of static sample points
    even_locs = get_evenly_spaced_locs(filter_size_static, samples_per_axis)

    # repeat sample points to sample for each point random
    grid_locs = np.repeat(even_locs, samples_per_axis)
    # [batch_size, num_locations]
    grid_locs_batch = np.tile(grid_locs, (batch_size, 1))

    rand_locs_batch = []
    for _ in range(batch_size):
        rand_locs = []
        for _ in range(samples_per_axis):
            # sample k random points per grid point
            rand_locs.append(np.random.choice(filter_size_rand, samples_per_axis, replace=False))

        # write sample locs per sample in list
        rand_locs_batch.append(np.hstack(rand_locs))

    # [batch_size, num_locations]
    rand_locs_batch = np.vstack(rand_locs_batch)

    # if time grid, return rand_locs for frequency axis
    if case=='time':
        return rand_locs_batch, grid_locs_batch
    else:
        return grid_locs_batch, rand_locs_batch


def get_vectors_from_maps_halfgrid(maps, x_locs, y_locs) -> torch.Tensor:
    
    # x_locs, y_locs shape: [batch_size, num_locs]
    
    batch_size, d, _, _ = maps.size()

    # [batch, d, num_locs]
    vectors = maps[np.arange(batch_size)[:,None], ..., x_locs, y_locs]

    # [batch * num_locs, d]
    vectors = vectors.transpose(-2, -1).reshape(-1, d)
    return vectors




def store_hook(module, input, output):
    # keep the output tensor gradient, even if it is not a leaf-tensor
    module.output = output
    output.retain_grad()


def get_intermediate(model, layer, input_batch, composite, class_idx, attr_batch_size=64, scaled_output=False) -> torch.Tensor:
    """
    Registers hook. Extracts activation maps and relevance maps at defined layer.
    """

    # TODO: hide progress bar when calling preprocessing from prototypes
    # possible with
    #if show_progress:
    #    iterable = tqdm(iterable, desc="Inner progress")
    #for item in iterable:


    input_batch_size = input_batch.size(0)

    # process data in smaller batches to avoid overloading the gpu by storing intermediate outputs
    num_batches = (input_batch_size + attr_batch_size - 1) // attr_batch_size 

    activation_maps, relevance_maps = [], []

    # register lrp hooks
    with Gradient(model, composite) as attributor:

        # register own store hook to save intermediate representations during forward pass (retain grad saves grad in backward pass)
        handles = [layer.register_forward_hook(store_hook)]

        with tqdm(total=input_batch_size, desc='Extracting activation and relevance maps') as pbar:

            for i in range(num_batches):

                batch = input_batch[i*attr_batch_size:min((i+1)*attr_batch_size, input_batch.size(0))]
                batch = batch.requires_grad_(True)

                # compute the relevance
                _, _ = attributor(batch, lrp_output_modifier(class_idx, scaled_output=scaled_output))

                activation_maps.append(layer.output.detach().to(device=batch.device))
                relevance_maps.append(layer.output.grad.detach().to(device=batch.device))

                layer.output = None

                pbar.update(batch.shape[0])

    # remove the store_hook
    for handle in handles:
        handle.remove()

    # extract activation and relevance maps 
    activation_maps = torch.concat(activation_maps, dim=0).requires_grad_(False)
    relevance_maps = torch.concat(relevance_maps, dim=0).requires_grad_(False)

    return activation_maps, relevance_maps


def compute_context_vectors(activation_vectors, relevance_vectors, epsilon) -> torch.Tensor:
    # add epsilon to avoid division by zero
    return relevance_vectors / (activation_vectors + epsilon)


def sample_spatial_location_old(map_size, num_locations) -> np.ndarray:

    idcs = np.random.choice(np.arange(map_size[0]*map_size[1]), num_locations, replace=False)
    #patch_idcs = idcs
    #x_locs = patch_idcs // int(map_size[0])
    #y_locs = patch_idcs % int(map_size[1])

    return idcs

    return x_locs, y_locs


def sample_spatial_location(batch_size, map_size, num_locations) -> np.ndarray:

    #idcs_batch = np.zeros((batch_size, num_locations), dtype=int)
    idcs_batch = np.zeros((batch_size, num_locations), dtype=int)

    # idcs for flattened feature maps
    for i in range(batch_size):
        idcs_batch[i, :] = np.random.choice(map_size[0]*map_size[1], num_locations, replace=False)

    return idcs_batch
    


def normalize_vectors(vectors) -> torch.Tensor:
    d = vectors.size()[-1]
    E = torch.sqrt(torch.mean(torch.square(vectors)))
    vectors_normalized = vectors / E / d**0.25
    return vectors_normalized


def get_vectors_from_maps_old(maps, idcs_batch) -> torch.Tensor:
    # idcs are for flattened feature maps
    
    # [batch, d, height, width]
    batch_size, d, _ = maps.size()
    # [batch, d, height * width]
    #maps = maps.reshape(batch_size, d, -1)

    # [batch, d, num_locs]
    vectors = maps[..., idcs_batch]

    # [batch * num_locs, d]
    vectors = vectors.transpose(-2, -1).reshape(-1, d)
    return vectors


def get_vectors_from_maps(maps, idcs_batch) -> torch.Tensor:
    # idcs are for flattened feature maps
    
    # [batch, d, height, width]
    batch_size, d, _, _ = maps.size()
    # [batch, d, height * width]
    maps = maps.reshape(batch_size, d, -1)

    # [batch, d, num_locs]
    vectors = maps[np.arange(batch_size)[:,None], :, idcs_batch]

    # [batch * num_locs, d]
    vectors = vectors.transpose(-2, -1).reshape(-1, d)
    return vectors


'''def get_songs(datapath, sample_class, split=None, excluded_folds=None, slice_length=6, num_chunks=4, to_mel=True, case='128_256', device=torch.device('cpu'), N=None, normalize=True):

    if case == 'toy':
        paths_to_songs = get_toy_samplelist(datapath, sample_class, split)
        if N is not None:
            random.shuffle(paths_to_songs)
            paths_to_songs = paths_to_songs[:N]

    elif case.endswith('filtered'):
        paths_to_songs = get_song_list_filtered(datapath, case, sample_class)
    
    else:
        paths_to_songs = get_songlist(datapath, sample_class, excluded_folds)

    data_batch = []
    paths_to_songs = paths_to_songs[:N]
    
    for path_to_song in tqdm(paths_to_songs, total=N):
        mels, _ = load_sample(path_to_song, slice_length=slice_length, num_chunks=num_chunks, to_mel=to_mel, case='128_256' if case.endswith('filtered') else case, 
                              normalize=normalize, audio_normalization='peak' if not normalize else None)
        mels = mels.detach()
        data_batch.extend(mels)
        
    data_batch_tensor = torch.stack(data_batch, dim=0)

    return data_batch_tensor, [path for path in paths_to_songs for _ in range(num_chunks)]'''


def get_songs_toy(datapath, sample_class, split=None, N=None):
    """
    Loading all samples of specific genre. Num chunks = 10. Then we shuffle the data and truncate the batch if defined.
    """

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

    # if N is specified we shuffle and truncate the batch of data
    #if N:
    #    data_batch_tensor, songs = shuffle_and_truncate_databatch(data_batch_tensor, songs, N)

    return data_batch_tensor, songs


# TODO: combine with get_sings_new_last from xai.prep

def get_songs_drsa(datapath, sample_class, excluded_folds=None, N=None, num_folds=5):
    """
    Loading all samples of specific genre. Num chunks = 10. Then we shuffle the data and truncate the batch if defined.
    """

    paths_to_songs = get_songlist(datapath, sample_class, excluded_folds, num_folds=num_folds)

    loader = Loader()
    num_chunks = 10
    
    data_batch = []
    songs = []

    # load samples as mel spectrograms
    for path_to_song in tqdm(paths_to_songs, total=len(paths_to_songs), desc="Loading samples from disk"):

        if path_to_song.endswith('hiphop.00036.wav') or path_to_song.endswith('hiphop.00038.wav'): continue         ################### TODO: delete this ####################

        mels = loader.load(path_to_song, num_chunks=num_chunks) 
        data_batch.extend(mels.detach())
        songs.extend([path_to_song for _ in range(num_chunks)])
        
    data_batch_tensor = torch.stack(data_batch, dim=0)

    # if N is specified we shuffle and truncate the batch of data
    if N:
        data_batch_tensor, songs = shuffle_and_truncate_databatch(data_batch_tensor, songs, N)

    return data_batch_tensor, songs
