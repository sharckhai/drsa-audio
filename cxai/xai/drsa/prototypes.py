import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import zennit

from cxai.xai.drsa.preprocessing import preprocess_data, get_songs_drsa
from cxai.xai.drsa.drsa import SubspaceOptimizer, objective_fn
from cxai.utils.utilities import HiddenPrints, round_down
from cxai.utils.constants import CLASS_IDX_MAPPER, CLASS_IDX_MAPPER_TOY


def get_prototypes_ts(model: nn.Module, 
                    layer_idx: int,
                    Q: torch.tensor, 
                    composite: zennit.composites, 
                    path_to_data: str, 
                    sample_class: str, 
                    n: int = 10,
                    N: int = None,
                    excluded_folds: int = None,
                    case='gtzan',
                    seed: int = 42, 
                    num_concepts: int = 4,
                    device=torch.device('cpu')):
    """
    Evaluates prototypical samples for the DRSA analysis. 
    A prototype is defined as a saple that achieves a high DRSA objective.
    """
    
    num_chunks = 10
    slice_length = 3
    class_idx_mapper = CLASS_IDX_MAPPER
    d_c = Q.size(0) // num_concepts

    # load full databatch
    data_batch, loaded_samples = get_songs_drsa(path_to_data, sample_class=sample_class, excluded_folds=excluded_folds)
    data_batch = data_batch.detach().requires_grad_(False).to(device)

    N = N if N else data_batch.size(0)

    local_gen = torch.Generator()
    local_gen = local_gen.manual_seed(seed)
    perm_mask = torch.randperm(data_batch.size(0), generator=local_gen)
    
    # keep track of startpoints because audios are sliced in smaller parts
    hop = round_down((29 - slice_length) / (num_chunks-1), 1)
    startpoints = (torch.arange(num_chunks)*hop).repeat(data_batch.size(0)//num_chunks)[perm_mask][:N]

    data_batch = data_batch[perm_mask][:N]
    loaded_samples = [loaded_samples[i] for i in perm_mask[:N]]

    max_obj = 0
    for i in tqdm(range(N//n), desc="Evaluating subsets"):

        subset = data_batch[i*n:(i+1)*n].clone().detach().to(device)

        with HiddenPrints():
            act_vecs, ctx_vecs = preprocess_data(model, subset, composite, layer_idx, class_idx_mapper[sample_class], case=case, device=device)

        act_vecs = act_vecs.reshape(-1, act_vecs.size(-1))
        ctx_vecs = ctx_vecs.reshape(-1, ctx_vecs.size(-1))

        # get obj value
        obj = SubspaceOptimizer.obj_val(act_vecs, ctx_vecs, Q, objective_fn, num_concepts=num_concepts, d_c=d_c)
        
        if obj > max_obj:
            max_obj = obj
            act_maps = act_vecs.clone().detach()
            ctx_maps = ctx_vecs.clone().detach()
            subset_idx = i

    songs = loaded_samples[subset_idx*n:(subset_idx+1)*n]
    startpoints = startpoints[subset_idx*n:(subset_idx+1)*n]

    return act_maps, ctx_maps, songs, startpoints


def get_prototypes_toy(model: nn.Module, 
                    layer_idx: int,
                    Q: torch.tensor, 
                    composite: zennit.composites, 
                    path_to_data: str, 
                    sample_class: str, 
                    n: int = 10,
                    N: int = None,
                    sample_rate: int = 16000,
                    excluded_folds: int = None,
                    case='gtzan',
                    seed: int = 42, 
                    num_concepts: int = 4,
                    device=torch.device('cpu')):
    
    class_idx_mapper = CLASS_IDX_MAPPER_TOY
    d_c = Q.size(0) // num_concepts

    # load full databatch
    data_batch, loaded_samples = get_songs_drsa(path_to_data, sample_class=sample_class, excluded_folds=excluded_folds)
    data_batch = data_batch.detach().requires_grad_(False).to(device)

    N = N if N else data_batch.size(0)

    local_gen = torch.Generator()
    local_gen = local_gen.manual_seed(seed)
    perm_mask = torch.randperm(data_batch.size(0), generator=local_gen)

    data_batch = data_batch[perm_mask][:N]
    loaded_samples = [loaded_samples[i] for i in perm_mask[:N]]

    max_obj = 0
    for i in tqdm(range(N//n), desc="Evaluating subsets"):

        subset = data_batch[i*n:(i+1)*n].clone().detach().to(device)

        with HiddenPrints():
            act_vecs, ctx_vecs = preprocess_data(model, subset, composite, layer_idx, class_idx_mapper[sample_class], case=case, device=device)

        act_vecs = act_vecs.reshape(-1, act_vecs.size(-1))
        ctx_vecs = ctx_vecs.reshape(-1, ctx_vecs.size(-1))

        # get obj value
        obj = SubspaceOptimizer.obj_val(act_vecs, ctx_vecs, Q, objective_fn, num_concepts=num_concepts, d_c=d_c)
        
        if obj > max_obj:
            max_obj = obj
            act_maps = act_vecs.clone().detach()
            ctx_maps = ctx_vecs.clone().detach()
            subset_idx = i

    songs = loaded_samples[subset_idx*n:(subset_idx+1)*n]
    startpoints = startpoints[subset_idx*n:(subset_idx+1)*n]

    return act_maps, ctx_maps, songs, startpoints

