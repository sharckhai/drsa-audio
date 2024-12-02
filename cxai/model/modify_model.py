import torch
import torch.nn as nn

class ProjectionModel(nn.Module):
    """Inserts the projection layers in a NN at the given layer_idx.
    Inserts forward proejction, backward and one filterlayer in the middle.
    
    NOTE: this is the solution for now, we inject those layers and can use the 
    model ininference mode while simultaneously extracting concept explanations.

    Other options:
    1. Only one extra layer (only  filter layer where we register the hook that seperates relevances 
    onto concepts). One custom LRP rule that initializes U, projects gradients onto h, seperates the 
    batch, projects back to a.
    2. DIfferent models for deployment so we dont have to seperate the batch within a backward hook,
    which is kind of haccky but safes a lot of memory and time.
    """

    def __init__(self, 
                 model: nn.Sequential, 
                 layer_idx: int, 
                 U: torch.Tensor, 
                 num_concepts: int, 
                 case: str = 'gtzan'
                 ) -> None:
        super().__init__()

        # set precalculated flat features between feature extractor and classifier
        self.num_flat_features = 2048 if case == 'gtzan' else 64

        assert layer_idx < len(model.features) and layer_idx > 0, \
            'layer_idx has to be in range 0 - len(model.features)'

        # init new feature extractor
        self.features = nn.Sequential()
        for idx, layer in enumerate(model.features.children()):
            if idx == layer_idx+1:
                self.features.add_module('projection', Projection(U, num_concepts))
                self.features.add_module('subspacefilter', SubspaceFilter())
                self.features.add_module('invprojection', InvProjection(U, num_concepts))
            self.features.add_module(str(idx), layer)

        # init new classifier
        self.classifier = nn.Sequential()
        for idx, layer in enumerate(model.classifier.children()):
            self.classifier.add_module(str(idx), layer)


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_flat_features)
        return self.classifier(x)


class SubspaceFilter(nn.Module):
    """We could filter relevances with the defined hook on the InvProjection Layer, 
    but for convenience we introduce a filter layer."""

    def __init__(self) -> None:
        super(SubspaceFilter, self).__init__()

    def forward(self, act_map) -> torch.Tensor:
        return act_map


class Projection(nn.Module):
    """Defines the projection from activations a onto subspaces h_k.
    
    - U has shape [d, d]
    - activation_maps have shape [b, d, filter_height, filter_width]
    - h has shape [batch, n, n_concepts, d_k]
    with n = filter_height * filter_width
    """

    def __init__(self, U: torch.Tensor, num_concepts: int) -> None:
        super(Projection, self).__init__()

        # U shape [d, d]
        self.U = U
        self.num_concepts = num_concepts
        self.d_k = self.U.size(0) // self.num_concepts


    def forward(self, act_map: torch.Tensor) -> torch.Tensor:

        # reshape act_vecs to [batch, n, d]
        act_vecs = act_map.view(act_map.size(0), act_map.size(1), -1).transpose(-2,-1).contiguous()

        # [batch, n, d]
        h = torch.matmul(act_vecs, self.U)

        # [batch, n, n_concepts, d_k]
        h = h.view(h.size(0), h.size(1), self.num_concepts, self.d_k)
        return h
    

class InvProjection(nn.Module):
    """Projects supspaces h_k onto recovered activations a'
    
    - U has shape [d, d]
    - h has shape [batch, n, n_concepts, d_k]
    - activation_maps have shape [b, d, filter_height, filter_width]
    """

    def __init__(self, U: torch.Tensor, num_concepts: int) -> None:
        super(InvProjection, self).__init__()

        self.U_inv = U.T
        self.num_concepts = num_concepts
        self.d = self.U_inv.size(0)
        self.d_k = self.d // self.num_concepts
        

    def forward(self, h: torch.Tensor) -> torch.Tensor:

        # h shape [batch, n, num_concepts, d_k]
        b, n, _, _ = h.size()

        # [batch, n, d]
        h = h.view(b, n, self.d)

        # inverse projection, [batch, n, d]
        a_ = torch.matmul(h, self.U_inv)
        
        # [batch, d, filter_height, filter_width]
        filter_height = filter_width = int(n**.5)
        a_ = a_.transpose(-2,-1).view(b, self.d, filter_height, filter_width).contiguous()
        return a_
    

class DifferentialLayer(nn.Module):
    """Defines special layer for LRP operation. See paper 2017, LRP overview."""

    def __init__(self, weights, bias, device=torch.device('mps')):
        super(DifferentialLayer, self).__init__()
        self.device = device
        
        # Initialize differential weights and biases
        # shape of weights: input_size x num_classes x num_classes
        weights = weights.T
        self.weights = (weights[:,:,None] - weights[:,None,:]).to(self.device)
        # shape of bias: num_classes x num_classes
        self.bias = (bias[None] - bias).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        # calculate log probability ratios
        out = torch.einsum('bi,ijk -> bjk', x, self.weights.to(x)) + self.bias.to(x)
        return out


class ReverseLogSumExp(nn.Module):
    """Defines special layer for LRP operation. See paper 2017, LRP overview."""

    def __init__(self):
        super(ReverseLogSumExp, self).__init__()
        #self.device = device

    def forward(self, x):

        # get exponentials
        exp_values = torch.exp(-x)

        # mask out the diagonal to ensure c != c' when summing up
        mask = 1 - torch.eye(exp_values.size(-1)).to(x)
        exp_values = exp_values * mask

        # negative log of sum
        out = -torch.log(torch.sum(exp_values, dim=-1))
        return out
    