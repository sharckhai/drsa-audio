from typing import Tuple
import warnings
warnings.filterwarnings("ignore", message="MPS: no support for int64 repeats mask, casting it to int32")

import torch
import torch.nn as nn
from zennit.attribution import Gradient
from zennit.core import Hook
from zennit.composites import Composite


class SubspaceHook(Hook):
    """This class implements a hook which masks the gradients (i.e., the relevances) during the backward pass.
    Since the explainer class clones each instance in the batch (num_concepts+1) times, this hook works by
    filtering the relevances of each clone in the batch through one concept subspace. This results in 
    relevance (num_concepts+1) different relevance heatmaps at the model input for each instance.
    The standard heatmap, and each concept-conditioned heatmap.
    
    Attributes:
        num_concepts (int): Number of subspaces that have been optimized.
        stabilizer (float): Used to avoid devisions with zero.
        device (torch.device): Device to perfrom computations.
    """
    
    def __init__(
        self, 
        num_concepts: int = 4, 
        stabilizer: float = 1e-7, 
        device: str | torch.device = torch.device('cpu')
    ) -> None:
        """
        Args:
            num_concepts (int, optional): Number of subspaces that have been optimized.
            stabilizer (float, optional): Used to avoid devisions with zero.
            device (str | torch.device, optional): Device to perfrom computations.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.stabilizer = stabilizer
        self.device = torch.device(device) if isinstance(device, str) else device

    def backward(self, module, grad_input, grad_output) -> Tuple[torch.Tensor,]:
        """Filters attributions through subspaces.
        
        Args:
            module (nn.Module): Module on which this hook is registered.
            grad_input (torch.Tensor): Input to the Module.
            grad_output (torch.Tensor): Gradient of the module where this hook is registered.

        Returns:
            Tuple[torch.Tensor,]: Modified gradient of the module.
        """
        grad_output, = grad_output
        batch, num_vecs, c, d_c = grad_output.size()
        grad_output = grad_output.view(-1, self.num_concepts+1, num_vecs, c, d_c)

        # mask grads of subspaces h_k, i.e. h_k*1, h_k' = 0 for all k' in K with k'!=k 
        grad_output[:, 1:] *= torch.eye(self.num_concepts, device=self.device)[None, :, None, :, None]
        grad_output = grad_output.view(batch, num_vecs, c, d_c)
        return grad_output,

    def copy(self):
        return self.__class__(
            num_concepts=self.num_concepts, 
            stabilizer=self.stabilizer, 
            device=self.device
        )
    

def compute_relevances(
    model: nn.Sequential, 
    input_batch: torch.Tensor, 
    composite: Composite,
    num_classes: int = None, 
    class_idx: int = None, 
    one_hot_encoded: bool = False
) -> torch.Tensor:
    """Performs the attribution for a batch or single input.

    NOTE: Input_batch is expected to contain samples of only one class OR samples of all classes.
    If batch contains samples of all classes, the batch has to be balanced and contain instances 
    of classes ordered in consecutive order.

    Args:
        model (nn.Sequential): model to perform attribution technique on.
        input_batch (torch.Tensor): Either samples of one class or all classes balanced in consecutive order.
        composite (Composite): A zennit composite with layer mappings and canonizer.
        class_idx (int, optional): Defines which class to attribute.
        num_classes (int, optional): TODO: check this.
        one_hot_encoded (bool, optional): If True, an output logit of 1 is used for relevance propagation.

    Returns:
        R (torch.Tensor): Tensor with relevances of the batched input samples. R has same size as input_batch:
    """
    # TODO: batch size only if one class attribution and subspace attrubution
    input_batch = input_batch.requires_grad_()
    # when entering we register the rule hooks
    with Gradient(model, composite) as attributor:
            # compute the relevance
            _, relevance_maps = attributor(
                input_batch, 
                lrp_output_modifier(
                    class_idx, 
                    num_classes, 
                    one_hot_encoded
                )
            )
    return relevance_maps


def lrp_output_modifier(
    class_idx: int = None, 
    num_classes: int = None, 
    one_hot_encoded: bool = False
) -> torch.Tensor:
    """Defines a function to modify the model output and determine which logit/logits to attribute.

    TODO: why class_idx and num_classes?

    Args:
        class_idx (int, optional): Defines which class to attribute
        num_classes (int, optional): TODO: check this
        one_hot_encoded (bool, optional): If True, an output logit of 1 is used for relevance propagation.

    Returns:
        output (torch.Tensor): Relevance heatmap.
    """

    assert class_idx is not None or num_classes is not None, 'Provide either class_idx to attribute \
        or samples_per_class to be able to build attribution mask for batch'

    if class_idx is not None:

        def extract_output_class(output):
            """Constructs a mask for class class_idx to attribute evidence (logits) of only this class."""
            # generate mask
            mask = torch.zeros_like(output)
            mask[..., class_idx] = 1

            if one_hot_encoded:
                return mask
            else:
                return output * mask
        return extract_output_class

    elif num_classes is not None:

        def attribute_all_classes(output):
            """Constructs a mask to attribute different class logits for different samples in batch. Batch
            is always expected to contain #samples_per_class class specific samples in consecutive order."""
            # generate mask, [batch_size, num_classes]
            mask = torch.repeat_interleave(torch.eye(num_classes).to(output), output.size(0) // num_classes, dim=0)

            if one_hot_encoded:
                return mask
            else:
                return output * mask
        return attribute_all_classes
    else:
        raise ValueError('Provide either class_idx to attribute or num_classes')
    