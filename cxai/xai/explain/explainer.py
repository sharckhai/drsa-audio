from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from zennit.rules import Pass, Epsilon, BasicHook
from zennit.composites import NameMapComposite, Composite

from cxai.model.modify_model import ProjectionModel
from cxai.xai.explain.attribute import compute_relevances
from cxai.xai.explain.attribute import SubspaceHook
from cxai.utils.constants import CLASS_IDX_MAPPER, CLASS_IDX_MAPPER_TOY


class HeatmapGenerator:
    """Class to generate explanations in input space.
    
    This class generatese the standard explanation and concept explantions as heatmaps 
    in input space for a given target class. This class assumes that subspaces have been 
    previously optimized.
    
    Attributes:
        info (Dict[str, Any]): The results. This dict containes 
        projectionmodel (nn.Sequential): NN model with inserted projectionmatrix that maps 
                                         activations onto relevant subspaces and back.
        composite (Composite): Zennit composite.
        num_concepts (int): Number of subspaces that were optimized.
        class_idx (int): Index of the class to attribute.
        num_classes (int): Total number of classes.
    """

    def __init__(
        self,
        model: nn.Sequential,
        U: torch.Tensor,
        name_map: List[Tuple[str, BasicHook]],
        sample_class: str,
        num_concepts: int = 4,
        layer_idx: int = 10,
        device: str | torch.device = torch.device('cpu'),
    ) ->  None:
        """
        Args:
            model (nn.Sequential): Model.
            U (torch.Tensor): Projection matrix.
            name_map (List[Tuple[str, BasicHook]]): Maps LRP rules to model layers.
                                                    The string defines the name of the layer.
            sample_class (str): Class to attribute.
            num_concepts (int, optional): Number of subspaces that were optimized.
            layer_idx (int, optional): Idx of layer were subspaces have been optimized.
            device (str | torch.device, optional): Device.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_concepts = num_concepts
        # infer case
        case = 'toy' if sample_class.endswith('1') or sample_class.endswith('2') else 'gtzan'
        class_idx_mapper = CLASS_IDX_MAPPER if case=='gtzan' else CLASS_IDX_MAPPER_TOY
        # get class index to attribute
        self.class_idx = class_idx_mapper[sample_class]
        self.num_classes = len(class_idx_mapper)
        # insert projection layers in NN model
        self.projectionmodel = ProjectionModel(model, layer_idx, U, self.num_concepts, case=case)
        # build up zennit composite to attribute relevances
        self.composite = get_class_composite(name_map, self.num_concepts, device=device)
        # init result dict
        self.info = {}

    def generate_subspace_heatmaps(
        self, 
        input_batch: torch.Tensor, 
        one_hot_encoded: bool = False, 
        concept_flipping: bool = False, 
        flip_all_classes: bool = False
    ) -> None:
        """Performs the two-step relevance attribution after training of the relevant subspaces.
        
        Samples in input batch get repeated num_concept+1 times to be able to attribute 
        standard heatmap and all subspace heatmaps in one signle pass.

        Args:
            input_batch (torch.Tensor): single input sample or batch
            one_hot_encoded (bool, optional): If True, an output logit of 1 is used for relevance propagation.
            concept_flipping (bool, optional): Flag for an evaluation method.
            flip_all_classes (bool, optional): Flag for evaluation functionalities. Attirbutes batch of samples form different classes
                                               through subspaces optimized for a single class.
        """
        # update result dict
        self.info['input'] = input_batch.cpu().numpy()

        # each instance is repeated num_concepts + 1 times
        input_batch = input_batch.to(self.device)
        repeated_input_batch = input_batch.repeat_interleave(self.num_concepts + 1, dim=0)

        # [batch*(n_concepts+1), channels, height, width]
        heatmaps = self.obtain_heatmaps(
            repeated_input_batch,
            one_hot_encoded,
            flip_all_classes
        ).squeeze()
        # reshape heatmaps
        heatmaps = heatmaps.view(-1, self.num_concepts+1, heatmaps.size(-2), heatmaps.size(-1))

        """# CASE: evaluation technique concept flipping. Has nothig to do 
        # with normal behaviour and use case of this class
        if concept_flipping and input.size(0) > 1:
            # only keep subspace heatmaps and get rid of standard heatmap
            # [batch, n_concepts, height, width]
            subspace_heatmaps = heatmaps[:, 1:].clone().detach()
            return subspace_heatmaps"""
        
        heatmaps = heatmaps.detach().cpu().numpy()
        # extract standard relevance heatmap and subspace heatmaps
        standard_heatmaps = heatmaps[:, 0:1]
        subspace_heatmaps = heatmaps[:, 1:]
        # sort subspaces according to descending relevance
        subspace_heatmaps, subspace_relevances, mask = self.sort_subspaces(subspace_heatmaps)

        # update result dict
        self.info['standard_heatmaps'] = standard_heatmaps
        self.info['standard_relevance'] = standard_heatmaps.sum(axis=(-2,-1)).flatten()
        self.info['subspace_heatmaps'] = subspace_heatmaps
        self.info['subspace_relevances'] = subspace_relevances
        self.info['mask'] = mask

    def obtain_heatmaps(
        self, 
        input_batch: torch.Tensor, 
        one_hot_encoded: bool = False, 
        flip_all_classes: bool = False
    ) -> torch.Tensor:
        """Calculates heatmaps of all instance in batch.
        
        Args:
            input_batch (torch.Tensor): Input batch of mel spectrograms.
            one_hot_encoded (bool, optional): If True, an output logit of 1 is used for relevance propagation.
            flip_all_classes (bool, optional): Flag for evaluation functionalities. Attirbutes batch of samples form different classes
                                               through subspaces optimized for a single class.

        Returns:
            heatmaps (torch.Tensor): All heatmaps.
        """
        return compute_relevances(
            self.projectionmodel, 
            input_batch, 
            self.composite, 
            one_hot_encoded=one_hot_encoded,
            class_idx=self.class_idx if not flip_all_classes else None, 
            num_classes=len(self.num_classes) if flip_all_classes else None
        )

    def sort_subspaces(
        self, 
        subspace_heatmaps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sorts the subspace heatmaps of each instance according to their relevance.
        
        Args:
            subspace_heatmaps (np.ndarray): Subspace heatmaps with sahpe [batch, n_concepts, height, width].
        
        Returns:
            tuple: A tuple containing:
                - subspace_heatmaps (np.ndarray): Heatmaps reordered.
                - subspace_relevances (np.ndarray): Total relevance per subspace and instance.
                - mask (np.ndarray): Perturbation mask defining the order of subspaces.
        """
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


def get_class_composite(
    name_map: List[Tuple[str, BasicHook]], 
    num_concepts: int, 
    device: str | torch.device = torch.device('cpu')
) -> Composite:
    """Builds a composite to attribute relevances filtered by subspaces.
    
    This function builds a zennit.composites.Composite with the provided name_map and 
    the projection matrix associated with the defined layer_idx and class.
    
    Args:
        name_map (List[Tuple[str, BasicHook]]): Maps LRP rules to model layers. The string defines the name of the layer.
        num_concepts (int): Nuber of subspaces that were optimized.
        device (str | torch.device, optional): Device.
    
    Returns:
        composite (Composite): Zennit composite for LRP.
    """
    # init subsapce seperation rule CXai
    name_map_copy = name_map.copy()
    # append special rules to new layers
    name_map_copy.append((['features.invprojection'], Epsilon()))
    name_map_copy.append((['features.subspacefilter'], SubspaceHook(num_concepts, device=device)))
    name_map_copy.append(([f'features.projection'], Epsilon()))
    return NameMapComposite(name_map=name_map_copy)


def compute_subspace_relevances(
    act_vecs: torch.Tensor, 
    ctx_vecs: torch.Tensor, 
    U: torch.Tensor, 
    n_concepts: int = 4
) -> torch.Tensor:
    """Computes subspace relevances for each instance. 

    Activation and context vectors have to have shape [batch, N, d].
    
    Args:
        act_vecs (torch.Tensor): Activation vectors. 
        ctx_vecs (torch.Tensor): Context vectors.
        U (torch.Tensor): Projection matrix.
        num_concepts (int): Nuber of subspaces that were optimized.
    
    Returns:
        x (torch.Tensor): 
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
