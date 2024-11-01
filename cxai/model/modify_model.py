import torch
import torch.nn as nn

class ProjectionModel(nn.Module):
    """
    Inserts extra layers in a NN.
    """

    def __init__(self, model: nn.Sequential, layer_idx, Q, num_concepts, case='gtzan') -> None:
        super().__init__()

        self.num_flat_features = 2048 if case == 'gtzan' else 64

        assert layer_idx < len(model.features) and layer_idx > 0, 'layer_idx has to be in range 0 - len(model.features)'

        self.features = nn.Sequential()
        for idx, layer in enumerate(model.features.children()):
            if idx == layer_idx+1:
                self.features.add_module('projection', Projection(Q, num_concepts))
                self.features.add_module('subspacefilter', SubspaceFilter())
                self.features.add_module('invprojection', InvProjection(Q, num_concepts))
            self.features.add_module(str(idx), layer)

        self.classifier = nn.Sequential()
        for idx, layer in enumerate(model.classifier.children()):
            self.classifier.add_module(str(idx), layer)


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_flat_features)
        return self.classifier(x)


class ProjectionModel2(nn.Module):

    def __init__(self, model: nn.Sequential, layer_idx, case='gtzan') -> None:
        super().__init__()

        self.num_flat_features = 2048 if case == 'gtzan' else 128

        assert layer_idx < len(model.features) and layer_idx > 0, 'layer_idx has to be in range 0 - len(model.features)'

        self.features = nn.Sequential()
        for idx, layer in enumerate(model.features.children()):
            if idx == layer_idx+1:
                #self.features.add_module('projection', Projection(Q, num_concepts))
                self.features.add_module('subspacefilter', SubspaceFilter())
                #self.features.add_module('invprojection', InvProjection(Q, num_concepts))
            self.features.add_module(str(idx), layer)

        self.classifier = nn.Sequential()
        for idx, layer in enumerate(model.classifier.children()):
            self.classifier.add_module(str(idx), layer)


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_flat_features)
        return self.classifier(x)
    


class SubspaceFilter(nn.Module):
    """
    We could filter relevances with the defined hook on the InvProjection Layer, but for convenience we introduce a filter layer.
    """

    def __init__(self) -> None:
        super(SubspaceFilter, self).__init__()

    def forward(self, act_map) -> torch.Tensor:
        return act_map



class Projection(nn.Module):

    """
    Outputs h
    """

    def __init__(self, U, num_concepts) -> None:
        super(Projection, self).__init__()

        # Q shape [d, d]
        # Q shape has to be: [k]
        # a shape D x f1 x f2

        self.U = U
        self.num_concepts = num_concepts
        self.d_c = self.U.size(0) // self.num_concepts


    def forward(self, act_map) -> torch.Tensor:

        # act_map [4, d, filter_size, filter_size]
        # reshape for mapping
        # [4, filter_size*filter_size, d]

        """
        1, 1, 128 and 128,128 -> 1,1,128 with hk being [:32], [32:64], ...
        i.e. reshape(batch, n, n_concepts, d_k): 1,1,4,32
        
        """
        # reshape act_vecs to [batch, n, d]
        act_vecs = act_map.view(act_map.size(0), act_map.size(1), -1).transpose(-2,-1).contiguous()

        # [batch, n, d]
        h = torch.matmul(act_vecs, self.U)
        
        # [batch, n_concepts, d_c, filter_size**2]
        h = h.view(h.size(0), h.size(1), self.num_concepts, self.d_c)

        return h
    

class InvProjection(nn.Module):

    """
    Outputs a'
    """

    def __init__(self, U, num_concepts) -> None:
        super(InvProjection, self).__init__()

        self.U_inv = U.T
        self.num_concepts = num_concepts
        self.d = self.U_inv.size(0)
        self.d_c = self.d // self.num_concepts
        

    def forward(self, h) -> torch.Tensor:

        # h shape [batch, num_vecs, num_concepts, d_c]
        b, num_vecs, _, _ = h.size()

        # [batch, num_vecs, d]
        h = h.view(b, num_vecs, self.d)

        # inverse projection, [batch, num_vecs, d]
        a_ = torch.matmul(h, self.U_inv)
        
        # [batch, d, filter_size, filter_size]
        filter_size = int(num_vecs**.5)
        a_ = a_.transpose(-2,-1).view(b, self.d, filter_size, filter_size).contiguous()

        return a_
    


class DifferentialLayer(nn.Module):
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
    

