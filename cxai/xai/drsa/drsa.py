import os
from tqdm import tqdm
import pickle
from pathilib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ortho_group


class SubspaceOptimizer:
    """This class performs the main optimization algorithm of DRSA. NOTE: provide arxiv link

    Attributes:
    -----------
    U (torch.tesor)
        Projection matrix U with shape (d,d). It is a 'concatenation' of submatrices U_k 
        with shape (d,d_k) with sum_k(d_k) = d.
    act_vecs (torch.tesor)
        Activtions vectors a at layer j where we want to inject the virual layers, 
        i.e., where we optimize for relevant subspaces.
    ctx_vecs (torch.tesor):
        Context vectors created from activation vectors and there associated relevances at layer j.
    num_concepts (int)
        The number of subspaces we want to optimize (this is K in the paper and above).
    obj_fn (callable)
        The DRSA objective.
    d_k (int)
        Subspace dimension as defined above.
    """

    def __init__(self, 
                 U: torch.Tensor, 
                 activation_vecs: torch.Tensor, 
                 context_vecs: torch.Tensor, 
                 path_to_model: str, 
                 num_concepts: int = 4, 
                 device: str | torch.device = torch.device('mps')
                 ) -> None:
        
        super(SubspaceOptimizer, self).__init__()

        assert num_concepts > 0, "num_concepts must be a positive number"
        assert U.size(0) % num_concepts == 0, "num_concepts must be a divisor of width (=height) of U"

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.path_to_model = path_to_model
        self.num_concepts = num_concepts
        self.d_k = U.size(0) // num_concepts

        # init model & data
        self.U = nn.Parameter(U.detach()).to(self.device)
        self.obj_fn = objective_fn
        self.act_vecs = activation_vecs.to(self.device)
        self.ctx_vecs = context_vecs.to(self.device)


    def run(self, steps=2000):
        """Main optimization loop.
        
        -----
        Args:
            steps (int): Training iterations.
        """

        obj_arr = []
        tbar = tqdm(range(1, steps+1))

        for it in tbar:
            self.U.requires_grad_(True)
            self.U.grad = None
            self.U.retain_grad()

            obj = SubspaceOptimizer.obj_val(self.act_vecs, self.ctx_vecs, self.U, self.obj_fn, self.num_concepts, self.d_k)
            obj.backward()

            #self.U.grad = project_grad(self.U.grad, self.U)
            self.U = orthogonalize(self.U + self.U.grad)
            obj_arr.append(obj.detach().cpu().numpy())
            if it%500 == 0:
                tbar.set_description('Obj value: %7.4f' % (obj_arr[-1]))
        
        # add final objective value to list
        #R_kn = self.model(self.act_vecs, self.ctx_vecs)
        obj = SubspaceOptimizer.obj_val(self.act_vecs, self.ctx_vecs, self.U, self.obj_fn, self.num_concepts, self.d_k)
        obj_arr.append(obj.detach().cpu().numpy())

        self.save_model()
        self.save_train_stats(obj_arr)


    @staticmethod
    def obj_val(act_vecs: torch.Tensor, 
                context_vecs: torch.Tensor, 
                U: torch.Tensor, 
                obj_fn: callable, 
                num_concepts: int, 
                d_c: int
                ) -> torch.Tensor:
        """Implementation of the DRSA objective."""

        # calculate projected vectors
        xa = torch.matmul(act_vecs, U)
        xc = torch.matmul(context_vecs, U) 

        # element wise multiplication (models response to activations)
        # [batch, n_concepts, d_c]
        x = torch.mul(xa, xc).view(-1, num_concepts, d_c)

        # it results contribution of each datapoint to each concept, shape: (num_concepts x batch)
        # [batch, n_concepts]
        return obj_fn(F.relu(torch.sum(x, dim=-1)))


    def save_train_stats(self, obj_arr: list):
        pd.DataFrame({'loss': obj_arr}).to_csv(os.path.join(self.path_to_model, 'train_stats.csv'))


    def save_model(self):
        # save projection matrix as pickle file
        with open(os.path.join(self.path_to_model, 'projection_matrix.pkl'), 'wb') as file:
            pickle.dump(self.U.data.cpu().numpy(), file)


def generalized_fmean(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    # take mean over last dimension (first step is mean over n, second is mean over concepts)
    return torch.pow(torch.mean(torch.pow(x, p), dim=0), 1/p)


@torch.no_grad() 
def project_grad(gradient: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Projects the gradient onto the constrained space (orthogonal constraint)
    """
    return gradient - torch.matmul(torch.matmul(U.T, gradient), U.T)


@torch.no_grad()
def orthogonalize(U: torch.Tensor) -> torch.Tensor:

    # UtU is real symmetric therefore eigenvalue decomposition can be applied (instead of svd)
    M = torch.matmul(U.T, U)
    S, V = torch.linalg.eigh(M.cpu().double())
    V = V.float()
    M_inv = torch.matmul(torch.matmul(V, torch.diag(1.0 / torch.sqrt(S.float()))), V.T).to(U)
    return torch.matmul(U, M_inv)


def objective_fn(input: torch.Tensor) -> torch.Tensor:
    ### input.size(): (batch_size x num_concepts)
    x = generalized_fmean(input, 2)
    x = generalized_fmean(x, 0.5)
    return x


class ProjectionMatrix(nn.Module):

    def __init__(self, U, num_concepts) -> None:
        super(ProjectionMatrix, self).__init__()

        self.U = nn.Parameter(U.requires_grad_(True))
        self.num_concepts = num_concepts
        self.d_c = self.U.size(0) // self.num_concepts


    def forward(self, act_vecs, context_vecs) -> torch.Tensor:

        # clculate projected vectors
        xa = torch.matmul(act_vecs, self.U)
        xc = torch.matmul(context_vecs, self.U) 

        # element wise multiplication (models response to activations)
        x = torch.mul(xa, xc).view(-1, self.num_concepts, self.d_c)

        # it results contribution of each datapoint to each concept, shape: (num_concepts x batch)
        return F.relu(torch.sum(x, dim=-1))
    

def main(activation_vecs, 
         context_vecs, 
         path_to_model, 
         num_concepts=4, 
         batch_size=64, 
         learning_rate=1, 
         steps=2000, 
         runs=3, 
         seed=42, 
         device=torch.device('mps')):

    print(f'DRSA starting on device {device} ...')

    np.random.seed(seed)

    # sample orthogonal matrix
    d = activation_vecs.size(-1)
    U = ortho_group.rvs(d)

    print('Orthogonal projection matrix U of size (%2d x %2d)' % (d,d))

    for run in range(1,runs+1):

        model_path = os.path.join(path_to_model, f'run{run}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # randomly permute U for each run
        mask = np.random.permutation(d)    
        U = torch.tensor(U[:, mask], dtype=activation_vecs.dtype, device=device)

        print('-'*20)
        print(f'Starting RUN {run}')

        drsa_optimizer = SubspaceOptimizer(U, 
                                       activation_vecs, 
                                       context_vecs, 
                                       model_path, 
                                       num_concepts=num_concepts, 
                                       batch_size=batch_size, 
                                       learning_rate=learning_rate, 
                                       device=device)
        
        drsa_optimizer.run(steps=steps)

    print('Done!')

