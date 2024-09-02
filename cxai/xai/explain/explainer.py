import os
import sys
import numpy as np
import torch
import torch.nn as nn

from zennit.rules import Pass, Epsilon
from zennit.composites import NameMapComposite

from cxai.model.modify_model import ProjectionModel
from cxai.xai.explain.attribute import compute_relevances
from cxai.xai.explain.attribute import SubspaceHook
from cxai.utils.constants import CLASS_IDX_MAPPER, CLASS_IDX_MAPPER_TOY


class HeatmapGenerator:
    r"""
    Class to generate standard heatmap and subspace heatmaps for a batch of samples of a target class. 
    """

    def __init__(self,
                 model: nn.Sequential,
                 U: torch.Tensor,
                 name_map: list, # in the form for zennit.composites
                 sample_class: str,
                 num_concepts: int = 4,
                 layer_idx: int = 10,
                 device: torch.device = torch.device('cpu'),
                 case: str = None,
                 ):
        
        self.device = device
        self.num_concepts = num_concepts
        self.sample_class = sample_class

        self.case = 'toy' if self.sample_class.endswith('1') or self.sample_class.endswith('2') else 'gtzan'
        self.CLASS_IDX_MAPPER = CLASS_IDX_MAPPER if self.case=='gtzan' else CLASS_IDX_MAPPER_TOY

        self.projectionmodel = ProjectionModel(model, layer_idx, U, self.num_concepts, case=self.case)
        self.composite = get_class_composite(name_map, self.num_concepts, device=device)
        self.class_idx = self.CLASS_IDX_MAPPER[self.sample_class]
        self.info = {}


    def generate_subspace_heatmaps(self, input: torch.Tensor, scaled_output=False, concept_flipping=False, flip_all_classes=False):
        r"""
        Samples in input batch get repeated num_concept+1 times to be able to attribute standard heatmap and all subspace heatmaps at once and in one pass.
        -----
        Args:
            input (torch.Tensor): single input sample or batch
        """
        # input sample is repeated num_concepts + 1 times
        repeated_input_batch = input.clone().detach().to(self.device).repeat_interleave(self.num_concepts + 1, dim=0)

        # [batch*(n_concepts+1), channels, height, width]
        heatmaps = self.obtain_heatmaps(repeated_input_batch, scaled_output=scaled_output, flip_all_classes=flip_all_classes).squeeze()
        heatmaps = heatmaps.view(-1, self.num_concepts+1, heatmaps.size(-2), heatmaps.size(-1))

        if concept_flipping and input.size(0) > 1: # why this?
            # only keep subspace heatmaps and get rid of standard heatmap
            # [batch, n_concepts, height, width]
            subspace_heatmaps = heatmaps[:, 1:].clone().detach()
            
            return subspace_heatmaps
        
        heatmaps = heatmaps.detach().cpu().numpy()
        # extract standard relevance heatmap and subspace heatmaps
        standard_heatmaps = heatmaps[:, 0:1]
        subspace_heatmaps = heatmaps[:, 1:]

        # sort subspaces according to descending relevance
        subspace_heatmaps, subspace_relevances, mask = self.sort_subspaces(subspace_heatmaps)

        self.info['input'] = input.cpu().numpy()
        self.info['standard_heatmaps'] = standard_heatmaps
        self.info['standard_relevance'] = standard_heatmaps.sum(axis=(-2,-1)).flatten()
        self.info['subspace_heatmaps'] = subspace_heatmaps
        self.info['subspace_relevances'] = subspace_relevances
        self.info['mask'] = mask


    def obtain_heatmaps(self, input_batch, scaled_output: bool = False, flip_all_classes: bool = False) -> torch.Tensor:

        return compute_relevances(self.projectionmodel, input_batch, self.composite, scaled_output=scaled_output,
                                  class_idx=self.class_idx if not flip_all_classes else None, 
                                  num_classes=len(self.CLASS_IDX_MAPPER) if flip_all_classes else None)

    
    def sort_subspaces(self, subspace_heatmaps):
        
        batch = subspace_heatmaps.shape[0]
        # sort subspaces according to relevance
        # [batch, n_concepts, height, width]
        subspace_relevances = subspace_heatmaps.sum(axis=(-2,-1)).squeeze()
        # [batch, n_concepts]
        mask = np.argsort(subspace_relevances, axis=-1)[..., ::-1]

        # sort heatmaps and relevances according to decreasing relevance
        subspace_heatmaps = subspace_heatmaps[np.arange(batch)[:, None], mask]
        subspace_relevances = subspace_relevances[np.arange(batch)[:, None], mask]

        return subspace_heatmaps, subspace_relevances, mask


def get_class_composite(name_map, num_concepts, device=torch.device('cpu')):
    r"""
    This function builds a NameMapComposite (zennit package) with the provided name_map the projection matrix associated with the defined layer_idx and class.
    The name_map defines which LRP rule is assigned to which layer of the model.
    """
    # init subsapce seperation rule CXai
    name_map_copy = name_map.copy()

    name_map_copy.append((['features.invprojection'], Epsilon()))
    name_map_copy.append((['features.subspacefilter'], SubspaceHook(num_concepts, device=device)))
    #name_map_copy.append(([f'features.filter'], sb()))
    name_map_copy.append(([f'features.projection'], Epsilon()))

    return NameMapComposite(name_map=name_map_copy)



def compute_subspace_relevances(act_vecs, ctx_vecs, U, n_concepts=4):
    r"""
    Computes subspace relevances for each sample in batch. Activation and context vectors have to be in shape [batch, N, d]
    """

    assert act_vecs.dim() < 4 or ctx_vecs.dim() < 4, 'Please provide act and ctx vectors reshaped to [batch, N, d]'
    
    # add virtual batch dimension if single sample is provided
    act_vecs = act_vecs if act_vecs.dim() == 3 else act_vecs.unsqueeze(0)
    ctx_vecs = ctx_vecs if ctx_vecs.dim() == 3 else ctx_vecs.unsqueeze(0)

    batch = act_vecs.size(0)
    d_c = U.size(0) // n_concepts

    # clculate projected vectors
    xa = torch.matmul(act_vecs, U)
    xc = torch.matmul(ctx_vecs, U)

    # element wise multiplication (models response to activations)
    x = torch.mul(xa, xc) #.view(-1, num_concepts, d_c)
    x = x.transpose(-2,-1).contiguous().view(batch, n_concepts, -1, d_c)
    
    return x.sum(-1).sum(-1)



