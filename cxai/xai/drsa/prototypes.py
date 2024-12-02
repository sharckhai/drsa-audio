from typing import Tuple, List

from tqdm import tqdm
import torch
import torch.nn as nn
from zennit.composites import Composite

from cxai.xai.drsa.preprocessing import preprocess_data, get_songs_drsa
from cxai.xai.drsa.drsa import SubspaceOptimizer, objective_fn
from cxai.utils.utilities import HiddenPrints, round_down
from cxai.utils.constants import CLASS_IDX_MAPPER, CLASS_IDX_MAPPER_TOY, AUDIO_PARAMS


def get_prototypes_ts(
    model: nn.Module, 
    layer_idx: int,
    U: torch.Tensor, 
    composite: Composite, 
    path_to_data: str, 
    sample_class: str,
    case: str = 'gtzan',
    num_concepts: int = 4,
    n: int = 10,
    N: int = None,
    excluded_folds: int = None,
    seed: int = 42,
    device: str | torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:
    """Evaluates prototypical samples for the DRSA analysis. 
    
    A prototype is defined as a saple that achieves a high DRSA objective.

    Args: 
        model (nn.Module): The NN.
        layer_idx (int): At which layer subspaces were optimized.
        Q (torch.Tensor): The projecrion matrix.
        composite (zennit.composites): LRP composite.
        path_to_data (str): Path to data root.
        sample_class (str): Which class should be attributed.
        case (str): 'gtzan' or 'toy'.
        num_concepts (int, optional): Number of subspaces.
        n (int, optional): Size of data subsets.
        N (int, optional): Total number of instances to load.
        excluded_folds (int, optional): Fold to exclude when loading data.
        seed (int, optional): Random seed.
        device (str | torch.device, optional): Device.

    Returns:
        tuple: A tuple containing:
            - prototypes_act_vecs (torch.Tensor): Activation vectors of prototypes.
            - prototypes_ctx_vecs (torch.Tensor): Context vectors of prototypes.
            - songs (List[str]): Songtitles.
            - startpoints (List[int]): Startpoints fo slices.
    """
    if isinstance(device, str): device = torch.device(device)

    # get class idx to class mapper
    class_idx_mapper = CLASS_IDX_MAPPER if case== 'gtzan' else CLASS_IDX_MAPPER_TOY
    d_c = U.size(0) // num_concepts

    # load data batch
    data_batch, loaded_samples = get_songs_drsa(
        path_to_data, 
        sample_class=sample_class, 
        excluded_folds=excluded_folds
    )
    data_batch = data_batch.to(device)

    # set number of instances
    N = N if N else data_batch.size(0)
    # create random generator
    local_gen = torch.Generator()
    local_gen = local_gen.manual_seed(seed)
    perm_mask = torch.randperm(data_batch.size(0), generator=local_gen)
    
    startpoints = None
    if case == 'gtzan':
        # extarct slices
        num_chunks = AUDIO_PARAMS['gtzan']['num_chunks']
        slice_length = AUDIO_PARAMS['gtzan']['slice_length']
        hop = round_down((29 - slice_length) / (num_chunks-1), 1)
        # keep track of startpoints because audios are sliced in smaller parts
        startpoints = (torch.arange(num_chunks)*hop)
        startpoints = startpoints.repeat(data_batch.size(0)//num_chunks)[perm_mask][:N]

    # shuffle data batch
    data_batch = data_batch[perm_mask][:N]
    loaded_samples = [loaded_samples[i] for i in perm_mask[:N]]

    max_obj = 0
    for i in tqdm(range(N//n), desc="Evaluating subsets"):

        # extract data subset
        subset = data_batch[i*n:(i+1)*n].clone().to(device)
        # hide prints
        with HiddenPrints():
            # extract activation and context vectors to compute subpsace relevances for an instance
            act_vecs, ctx_vecs = preprocess_data(
                model, 
                subset, 
                composite, 
                layer_idx, 
                class_idx_mapper[sample_class], 
                case=case, 
                device=device
            )
        # reshape for objective function
        act_vecs = act_vecs.reshape(-1, act_vecs.size(-1))
        ctx_vecs = ctx_vecs.reshape(-1, ctx_vecs.size(-1))

        # get obj value
        obj = SubspaceOptimizer.obj_val(
            act_vecs, 
            ctx_vecs, 
            U,
            objective_fn, 
            num_concepts=num_concepts, 
            d_c=d_c
        )
        # subset group achieves the highest relevance
        if obj > max_obj:
            max_obj = obj
            prototypes_act_vecs = act_vecs.clone()
            prototypes_ctx_vecs = ctx_vecs.clone()
            subset_idx = i

    # extract songs and startpoints of best subset
    songs = loaded_samples[subset_idx*n:(subset_idx+1)*n]
    startpoints = startpoints[subset_idx*n:(subset_idx+1)*n]
    return prototypes_act_vecs, prototypes_ctx_vecs, songs, startpoints
