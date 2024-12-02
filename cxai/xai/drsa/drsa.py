import os
from tqdm import tqdm
import pickle
from pathilib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ortho_group


class SubspaceOptimizer:
    """Main optimization algorithm of DRSA. 
    
    This class trains a random orthogonal matrix U, by optimizing it via gradient ascent. After each
    gradient ascent step, U is orthogonalized. During training, U is divided into multiple matrices U_k
    that map the training data (activation and context vectors) onto latent subspaces h_k.
    Arxiv link: https://arxiv.org/abs/2212.14855
    NOTE: A big part of the DRSA optimization procedure is performed in preprocessing.py, to extract the 
    training data.

    Attributes:
        U (torch.Tesor): Projection matrix U with shape (d,d). It is a 'concatenation' of 
                         submatrices U_k with shape (d,d_k) with sum_k(d_k) = d.
        act_vecs (torch.Tesor): Activtions vectors a at layer j where we want to inject the 
                                virual layers, i.e., where we optimize for relevant subspaces.
        ctx_vecs (torch.Tesor): Context vectors created from activation vectors and there 
                                associated relevances at layer j.
        num_concepts (int): The number of subspaces we want to optimize (this is K in the paper).
        obj_fn (callable): The DRSA objective.
        d_k (int): Subspace Dimension as defined above.
        device (torch.device): Device.
    """

    def __init__(
        self, 
        U: torch.Tensor, 
        activation_vecs: torch.Tensor, 
        context_vecs: torch.Tensor, 
        path_to_model: str, 
        num_concepts: int = 4, 
        device: str | torch.device = torch.device('mps')
    ) -> None:
        """Init the optimizer class.
        
        Args:
            U (torch.Tesor): Random orthogonal matrix.
            activation_vecs (torch.Tesor): Activtions vectors a at layer j where we optimize 
                                           for relevant subspaces.
            context_vecs (torch.Tesor): Context vectors created from activation vectors and 
                                        there associated relevances at layer j.
            path_to_model (str): Where to save the model (i.e., U).
            num_concepts (int, optional): The number of subspaces we want to optimize.
            device (str | torch.device, optional): Device.
        """
        super(SubspaceOptimizer, self).__init__()

        assert num_concepts > 0, "num_concepts must be a positive number"
        assert U.size(0) % num_concepts == 0, "num_concepts must be a divisor of width (=height) of U"

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.path_to_model = path_to_model
        self.num_concepts = num_concepts
        # calculate dimension of subspaces h_k
        self.d_k = U.size(0) // num_concepts
        # init projection model
        self.U = nn.Parameter(U.detach()).to(self.device)
        self.obj_fn = objective_fn
        # set training data as class attribute
        self.act_vecs = activation_vecs.to(self.device)
        self.ctx_vecs = context_vecs.to(self.device)

    def run(self, steps: int = 2000) -> None:
        """Main optimization loop.
        
        Args:
            steps (int, optional): Training iterations.
        """
        obj_arr = []
        tbar = tqdm(range(1, steps+1))
        for it in tbar:

            self.U.requires_grad_(True)
            self.U.grad = None
            self.U.retain_grad()

            # compute objective value of DRSA
            obj = SubspaceOptimizer.obj_val(
                self.act_vecs, 
                self.ctx_vecs,
                self.U, 
                self.obj_fn, 
                self.num_concepts, 
                self.d_k
            )
            # backpropagate gradients to matrix entries
            obj.backward()
            # adjust matrix entries and orthogonalize new projection matrix
            self.U = orthogonalize(self.U + self.U.grad)
            # log training data
            obj_arr.append(obj.detach().cpu().numpy())
            if it%500 == 0:
                tbar.set_description('Obj value: %7.4f' % (obj_arr[-1]))
        
        # calculate final objective value
        obj = SubspaceOptimizer.obj_val(
            self.act_vecs, 
            self.ctx_vecs,
            self.U, 
            self.obj_fn, 
            self.num_concepts, 
            self.d_k
        )
        obj_arr.append(obj.detach().cpu().numpy())
        # save model and training stats
        self.save_model()
        self.save_train_stats(obj_arr)

    @staticmethod
    def obj_val(
        act_vecs: torch.Tensor, 
        context_vecs: torch.Tensor, 
        U: torch.Tensor, 
        obj_fn: callable, 
        num_concepts: int, 
        d_k: int
    ) -> torch.Tensor:
        """Implementation of the DRSA objective.

        For more information see the thesis report or the DRSA paper.

        Args:
            activation_vecs (torch.Tesor): Activtions vectors a at layer j where we optimize 
                                           for relevant subspaces.
            context_vecs (torch.Tesor): Context vectors created from activation vectors and 
                                        there associated relevances at layer j.
            U (torch.Tensor): Orthogonal projection matrix.
            obj_fn (callable): The DRSA objective.
            d_k (int): Subspace dimension as defined above.

        Returns:
            objective (torch.Tensor): Objective value.
        """
        # calculate projected vectors
        xa = torch.matmul(act_vecs, U)
        xc = torch.matmul(context_vecs, U) 
        # element wise multiplication (models response to activations)
        # [batch, n_concepts, d_c]
        x = torch.mul(xa, xc).view(-1, num_concepts, d_k)
        # it results contribution of each datapoint to each concept, shape: (num_concepts x batch)
        # [batch, n_concepts]
        return obj_fn(F.relu(torch.sum(x, dim=-1)))

    def save_train_stats(self, obj_arr: List[np.ndarray]) -> None:
        """Log train stats to .csv-file.
        
        Args:
            obj_arr (List[np.ndarray]): Objectives of each training iteration.
        """
        pd.DataFrame({'loss': obj_arr}).to_csv(os.path.join(self.path_to_model, 'train_stats.csv'))

    def save_model(self) -> None:
        """Save model, i.e., the projection matrix U as .pkl-file."""
        with open(os.path.join(self.path_to_model, 'projection_matrix.pkl'), 'wb') as file:
            pickle.dump(self.U.data.cpu().numpy(), file)


def generalized_fmean(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Implementation of a gerneralized F-mean with function F(t)=t^p.
    
    Args:
        x (torch.Tensor): Tensor to compute the mean over.
        p (float): Power of t.

    Returns:
        mean (torch.Tensor): Generalized fmean over x with power p and function t^p.
    """
    # take mean over last dimension (first step is mean over n, second is mean over concepts)
    return torch.pow(torch.mean(torch.pow(x, p), dim=0), 1/p)


@torch.no_grad() 
def project_grad(gradient: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Projects the gradient onto the constrained space (orthogonal constraint).
    
    This function is currently unused.
    
    Args:
        gradient (torch.Tensor): Gradient wrt. U.
        U (torch.Tensor): Orthogonal projection matrix.

    Return:
        Projected gradient onto the constrained space.
    """
    return gradient - torch.matmul(torch.matmul(U.T, gradient), U.T)


@torch.no_grad()
def orthogonalize(U: torch.Tensor) -> torch.Tensor:
    """Orthogonalize a squared matrix with eigenvalue decomposition.
    Matrix is orthogonalized with U <- U(U.T@U)^(-0.5).
    
    Args:
        U (torch.Tensor): Square matrix that should be orhtogonalized.

    Returns:
        U (torch.Tensor): Orthogonalized input matrix.
    """
    # UtU is real symmetric therefore eigenvalue decomposition can be applied (instead of svd)
    UtU = torch.matmul(U.T, U)
    # perform eigenvalue decomposition
    # NOTE: when using 'mps' GPU, switch to cpu().double(). Not necessary with cuda.
    S, V = torch.linalg.eigh(UtU.cpu().double())
    V = V.float()
    # construct inverse square-root of UtU
    UtU_inv = torch.matmul(torch.matmul(V, torch.diag(1.0 / torch.sqrt(S.float()))), V.T).to(U)
    # perform orthogonalizatzion of U by multiplying with the inverse square-root
    return torch.matmul(U, UtU_inv)


def objective_fn(input: torch.Tensor) -> torch.Tensor:
    """Computes the DRSA objective. 
    
    Performs soft-min pooling over concepts and soft-max pooling over datapoints.
    
    Args:
        input (torch.Tensor): Relevance contribution of each datapoint to each subspace (concepts).

    Retirns:
        objective (torch.Tensor): DRSA objective.
    """
    # input size: [batch_size x num_concepts]
    x = generalized_fmean(input, 2)
    x = generalized_fmean(x, 0.5)
    return x
    

def main(
    activation_vecs: torch.Tensor, 
    context_vecs: torch.Tensor, 
    model_root: str, 
    num_concepts: int = 4, 
    steps: int = 2000, 
    runs: int = 3, 
    seed: int = 42, 
    device: str | torch.device = torch.device('mps')
) -> None:
    """Performs several training runs with different random seeds.
    
    Args:
        activation_vecs (torch.Tesor): Activtions vectors a at layer j where we optimize 
                                           for relevant subspaces.
        context_vecs (torch.Tesor): Context vectors created from activation vectors and 
                                    there associated relevances at layer j.
        model_root (str): Path to the model root folder.

        steps (int, optional): Training iterations (steps of gradient ascent).
        runs (int, optional): Train different models with different random seeds.
        seed (int, optional): Random seed.
        device (str | torch.device): Device.
    """
    np.random.seed(seed)
    if isinstance(device, str): device = torch.device(device)

    print(f'Starting DRSA training on device {device} ...')

    # sample orthogonal matrix
    d = activation_vecs.size(-1)
    U = ortho_group.rvs(d)

    print('Orthogonal projection matrix U of size (%2d x %2d)' % (d,d))

    # train different models
    for run in range(1,runs+1):
        # create model root and different path for each run
        model_path = os.path.join(model_root, f'run{run}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # randomly permute U for each run
        mask = np.random.permutation(d)    
        U = torch.tensor(U[:, mask], dtype=activation_vecs.dtype, device=device)

        print('-'*20, f'\nStarting RUN {run}')

        # init optimizer
        drsa_optimizer = SubspaceOptimizer(
            U, 
            activation_vecs, 
            context_vecs, 
            model_path, 
            num_concepts=num_concepts, 
            device=device
        )
        # train the model
        drsa_optimizer.run(steps=steps)
    
    print('Done!')