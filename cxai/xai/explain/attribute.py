import torch
import warnings
warnings.filterwarnings("ignore", message="MPS: no support for int64 repeats mask, casting it to int32")

from zennit.attribution import Gradient
from zennit.core import BasicHook, Hook, stabilize


class SubspaceHook(Hook):
    r"""
    In the forward pass, the explainer-class duplicates every sample k+1 times. This function is 
    a hook which masks the gradients (i.e., the relevances) during the backward pass.
    It results in each sample carrying the relevance of a single concept k. The first sample in each 
    minibatch of identical samples (k+1 samples) is not filtered and therefore carries the total relevance 
    (standard relevance).
    """

    def __init__(self, num_concepts=4, stabilizer=1e-7, device=torch.device('cpu')):
        super().__init__()
        self.num_concepts = num_concepts
        self.stabilizer = stabilizer
        self.device = device

    def backward(self, module, grad_input, grad_output):

        grad_output, = grad_output

        batch, num_vecs, c, d_c = grad_output.size()
        grad_output = grad_output.view(-1, self.num_concepts+1, num_vecs, c, d_c)

        # mask grads of subspaces h_k, i.e. h_k*1, h_k' = 0 for all k' in K with k'!=k 
        grad_output[:, 1:] *= torch.eye(self.num_concepts, device=self.device)[None, :, None, :, None]
        grad_output = grad_output.view(batch, num_vecs, c, d_c)

        return grad_output,


    def copy(self):
        return self.__class__(num_concepts=self.num_concepts, stabilizer=self.stabilizer, device=self.device)
    
    

def compute_relevances(model, input_batch, composite, num_classes=None, class_idx=None, scaled_output=False, attr_batch_size=64):
    r"""
    Performs the attribution for a batch or single input.

    NOTE: Input_batch is expected to contain samples of only one class OR samples of all classes.
    If batch contains samples of all classes, the batch has to be balanced and contain samples in consecutive order.
    To get evidence for a single class provide class_idx.
    To get evidence for different classes provide num_classes. 
    Dont provide both. Assertion is performed in lrp_output_modifier().
    -----
    Args:
        model       (nn.Sequential): model to perform attribution technique on
        input_batch (torch.Tensor): Either samples of one class or all classes balanced in consecutive order.
        composite   (zennit.composites): A zennit composite with layer mappings and canonizer.
        n_classes   (int): Number of classes.
    Returns:
        R           (torch.Tensor): Tensor with relevances of the batched input samples. R has same size as input_batch:
    """

    # TODO: batch size only if one class attribution and subspace attrubution

    input_batch = input_batch.requires_grad_()

    # when entering we register the rule hooks
    with Gradient(model, composite) as attributor:
            # compute the relevance
            _, relevance_maps = attributor(input_batch, lrp_output_modifier(class_idx, num_classes, scaled_output))

    return relevance_maps



def lrp_output_modifier(class_idx: int = None, num_classes: int = None, scaled_output: bool = False):
    """
    Defines a function to modify the model output and determine which logit/logits to attribute.
    -----
    Args:
        int  (class_idx): Defines which class to attribute
        bool (scaled_output): Determines if the output logit or a one-hot-encoded vector is used as output relevance.
                            If true the function returns one-hot-encoded array fro the class to attribute.
    """

    assert class_idx is not None or num_classes is not None, 'Provide either class_idx to attribute or samples_per_class to be able to build attribution mask for batch'

    if class_idx is not None:

        def extract_output_class(output):
            """
            Constructs a mask for class class_idx to attribute evidence (logits) of only this class
            """
            # generate mask
            mask = torch.zeros_like(output)
            mask[..., class_idx] = 1

            if scaled_output:
                return mask
            else:
                return output * mask

        return extract_output_class


    elif num_classes is not None:

        def attribute_all_classes(output):
            """
            Constructs a mask to attribute different class logits for different samples in batch.
            Batch is always expected to contain #samples_per_class class specific samples in consecutive order.
            """
            # generate mask, [batch_size, num_classes]
            mask = torch.repeat_interleave(torch.eye(num_classes).to(output), output.size(0) // num_classes, dim=0)

            if scaled_output:
                return mask
            else:
                return output * mask

        return attribute_all_classes
    
    else:
        raise ValueError('Provide either class_idx to attribute or num_classes')
    

