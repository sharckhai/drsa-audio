import torch
from torch import mps
import warnings
warnings.filterwarnings("ignore", message="MPS: no support for int64 repeats mask, casting it to int32")

from zennit.attribution import Gradient
from zennit.core import Stabilizer
from zennit.core import BasicHook, Hook, stabilize
from zennit.rules import NoMod


class SubspaceFilter(Hook):
    '''
    Note: 
    Has to be registered to Activation function, where concepts were extracted, or to extra Filter layer.
    It is possible to combiine this rule with some other rule to register it to the convolutional layer, 
    but this is the most simple solution.

    Description:
    This hook filters gradient, i.e. relevance, by prjecting onto subspaces and masking gradients.
    Expects the individual samples in the batch to be repeated (num_concepts+1) times, to generate standard heatmap + subspace heatmaps.
    Same instances are consecutive in the batch.

    # grad out is of shape
    [batch*(num_concepts+1), d, fx, fy]

    1. gets projected onto h with U
    2. gets reshape to [batch, num_concepts+1, N, num_concepts, d_c], 
        with N=fx*fx (filtersize. i.e. num of vectors), and d_c being subspace dimension
    3. The replicas of eacah instance, of whihc num_concept replicas are in the batch, get masked, to eahc only 
        retain relevances associated to one single subspace
    4. All data gets projected back to activaiton space with U.T
    5. Batch is reshaped to fit the shape of activation maps

    This works, as the gradient of a' to h, is the projection matrix U. Instead of inserting layers and using the gradient for masking,
    we can simple map gradients, i.e. relevances, onto subspaces, wxtrat their 'share' there and map them back

    '''

    def __init__(self, U, num_concepts=4, stabilizer=1e-7, device=torch.device('cpu')):
        super().__init__()
        self.U = U
        self.num_concepts = num_concepts
        self.stabilizer = stabilizer
        self.device = device

    def backward(self, module, grad_input, grad_output):

        grad_output, = grad_output

        batch, d, fx, fy = grad_output.size()
        grad_output = grad_output.view(batch, d, -1).transpose(-2,-1).contiguous()

        # mapping onto subspaces, shape [batch, N, d]
        grad_out_k = torch.matmul(grad_output, self.U)

        # reshape and mask
        grad_out_k = grad_out_k.view(-1, self.num_concepts+1, fx*fy, self.num_concepts, d//self.num_concepts)

        # mask grads of subspaces h_k, i.e. h_k*1, h_k' = 0 for all k' in K with k'!=k 
        grad_out_k[:,1:] *= torch.eye(self.num_concepts, device=self.device)[None, :, None, :, None]

        # backprojection to activation space
        grad_out_k = grad_out_k.view(batch, fx*fy, d) @ self.U.T
        # reshape to fit shape of activation maps
        grad_out_filtered = grad_out_k.transpose(-2,-1).contiguous().view(batch, d, fx, fy)

        return grad_out_filtered,


    def copy(self):
        return self.__class__(U=self.U, num_concepts=self.num_concepts, stabilizer=self.stabilizer, device=self.device)
    


class SubspaceHook(Hook):
    '''
    To attach to INVERSE PROJECTION!

    grad output, i.e., relevances, shape: [b, nv, c, d_c].

    since in each sample batch is repeated num_concepts+1 time, we need to reshape and mask gradients to only retain relevances per concept

    reshape to [num_concepts+1, -1, nv, c, d_c]

    divide, i.e. extract standard relevance

    standard_rel = [1, batch, nv, c, d_c]
    concept_grads = [num_concepts, batch, nv, c, d_c]

    mask concept grads, i.e. h_k*1, h_k' = 0 for all k' in K with k'!=k

    filtered_concept_grads = [num_concepts, batch, nv, c, d_c]

    full_batch = torch.stack(standard_rel, concept_grads).reshape(b, nv, c, d_c)

    then proceed with normal gradient flow

    


    This function expects batched input of size  1 + num_concepts of the same sample. 
    This Hook returns full R at this layer and each subpace filtered R.

    a             --p->        h         --ip->       a'
    [b, d, f, f]        [b, nv, c, d_c]          [b, d, f, f]

    '''

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
    """
    Performs the attribution for a batch or single input.

    NOTE:
    Input_batch is expected to contain samples of only one class OR samples of all classes.
    If batch contains samples of all classes, the batch has to be balanced and contain samples in consecutive order.

    To get evidence for a single class provide class_idx.
    To get evidence for different classes provide num_classes. 
    Dont provide both. Assertion is performed in lrp_output_modifier().

    Args:
    -----
    model: nn.Sequential
        model to perform attribution technique on
    input_batch: torch.Tensor
        Either samples of one class or all classes balanced in consecutive order.
    composite: zennit.composites
        A zennit composite with layer mappings and canonizer.
    n_classes: int
        Number of classes.

    Returns:
    -------
    R: torch.Tensor
        Tensor with relevances of the batched input samples. R has same size as input_batch:
    """

    # TODO: batch size only if one class attribution and subspace attrubution

    input_batch = input_batch.requires_grad_()

    """ # process data in smaller batches to avoid overloading the gpu by storing intermediate outputs
    num_batches = (input_batch.size(0) + attr_batch_size - 1) // attr_batch_size 

    relevance_maps = []

    # when entering we register the rule hooks
    with Gradient(model, composite) as attributor:

        for i in range(num_batches):

                batch = input_batch[i*attr_batch_size:min((i+1)*attr_batch_size, input_batch.size(0))]
                batch = batch.requires_grad_(True)

                # compute the relevance
                _, relevance_maps_batch = attributor(batch, lrp_output_modifier(class_idx, num_classes, scaled_output))


                relevance_maps.append(relevance_maps_batch)

                #mps.empty_cache()
                torch.cuda.empty_cache()

    return torch.concat(relevance_maps, dim=0).requires_grad_(False)"""

    # when entering we register the rule hooks
    with Gradient(model, composite) as attributor:

            # compute the relevance
            _, relevance_maps = attributor(input_batch, lrp_output_modifier(class_idx, num_classes, scaled_output))

    return relevance_maps



def lrp_output_modifier(class_idx: int = None, num_classes: int = None, scaled_output: bool = False):
    """
    Defines a function to modify the model output and determine which logit/logits to attribute.
    
    Args:
    -----
    int: class_idx
        Defines which class to attribute
    bool: scaled_output
        Determines if the output logit or a one-hot-encoded vector is used as output relevance
        If true the function returns one-hot-encoded array fro the class to attribute.

    Returns:
    -------
    funtion
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


    elif num_classes:

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
    

