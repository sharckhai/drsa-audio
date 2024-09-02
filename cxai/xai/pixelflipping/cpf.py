import os
import random
import numpy as np
import torch
import pickle
from tqdm import tqdm
from scipy.stats import ortho_group

from zennit.rules import WSquare, Gamma, Epsilon

from cxai.model.create_model import VGGType
from cxai.xai.pixelflipping.core import Flipper
from cxai.xai.pixelflipping.pf import PixelFlipping
from cxai.xai.pixelflipping.prep import get_data_main
from cxai.xai.explain.explainer import HeatmapGenerator
from cxai.utils.evaluation import get_best_run
from cxai.utils.constants import CLASS_IDX_MAPPER, CLASS_IDX_MAPPER_TOY



def concept_flipping(model, input_batch, name_map, layer_idx, path_to_U, num_concepts=4, standard_r=False, case=None, device=torch.device('cpu')):
    r"""
    Function to perform concept patch flipping if model and optimized projection matrices are provided. 
    Class cf_core performs flipping, this function loads the projection matrix and calculates mean scores.
    -----    
    Args:
        model       (nn.Sequential): Neural network to perform concept patch flipping on
        input_batch (torch.Tensor): Balanced tensor of datasamples of each class in consecutiove order
        name_map    (dict): name_map fpr zennit composite
        layer_idx   (int): Layer index to insert projection matrix Q
        path_to_Qs  (str): Path to root folder that contains the optimized projection matrices
    Returns:
        aupc_scores_per_instance        (np.ndarray): AUPC scores
        averaged_pertubed_predictions   (np.ndarray): Averaged prediction logits for plotting
        flips_per_perturbation_step     (np.ndarray): Patches flipped at each step
    """

    if isinstance(input_batch, np.ndarray): input_batch = torch.tensor(input_batch)

    class_idx_mapper = CLASS_IDX_MAPPER if case != 'toy' else CLASS_IDX_MAPPER_TOY

    input_batch = input_batch.to(device)
    samples_per_class = input_batch.size(0) // len(class_idx_mapper.keys())
    subspace_heatmaps = []

    # loop that aggregates subspace heatmaps of each genre
    for i, genre in tqdm(enumerate(class_idx_mapper.keys()), unit_scale=samples_per_class, total=len(class_idx_mapper.keys())):

        # extract samples of each class in each iteration
        class_batch = input_batch[i*samples_per_class:(i+1)*samples_per_class].clone().detach().to(device)

        # load projection matrix
        U = load_projection_matrix(genre, layer_idx, path_to_U, device=device)

        # instanciate heatmap generator
        generator = HeatmapGenerator(model, U, name_map, sample_class=genre, num_concepts=num_concepts, 
                                     layer_idx=layer_idx, case=case, device=device)
        class_subspace_heatmaps = generator.generate_subspace_heatmaps(class_batch, concept_flipping=True)
        subspace_heatmaps.append(class_subspace_heatmaps)


    # add all genre specific subsopaces into one tensor
    subspace_heatmaps = torch.cat(subspace_heatmaps, dim=0)
    subspace_heatmaps = np.array(subspace_heatmaps.cpu())

    #seperability = np.max(subspace_heatmaps,1).sum((-2,-1)) - np.max(subspace_heatmaps.sum((-2,-1)), 1)

    # init flipper
    flipper = Flipper(perturbation_size=16, device=device)
    def forward_func(input_batch): return model(input_batch)

    # flip whole batch (all classes)
    #with HiddenPrints():
    aupc_scores_per_instance, averaged_pertubed_predictions, flips_per_perturbation_step = flipper(forward_func, input_batch, subspace_heatmaps)
 
    return aupc_scores_per_instance, averaged_pertubed_predictions, flips_per_perturbation_step

     
def interclass_concept_flipping(model, input_batch, name_map, path_to_U, case=None, standard_r=False, toy=False, num_concepts=4, device=torch.device('cpu')):
    r"""
    Goal: 
    We want to know disentanglement power of the class specific concepts of a class.
    Since concepts of different classes look similar at first sight.

    Solution:
    Generate subspace heatmaps for samples of a target class by aggregating subspace heatmaps generated with Q of each genre.
    We hope disentanglement, i.e. AUPC, is smallest for Q of target class.
    -----
    Args:
        model       (nn.Sequential): Neural network to perform concept patch flipping on
        input_batch (torch.Tensor): Balanced tensor of datasamples of each class in consecutiove order
        name_map    (dict): name_map fpr zennit composite
        layer_idx   (int): Layer index to insert projection matrix Q
        path_to_Qs  (str): Path to root folder that contains the optimized projection matrices
    Returns:
        aupcs_per_class (np.ndarray): Shape: [n_classes, n_classes]
                                      - Columns correspond to genre that got attributed
                                      - Rows correspond to genre of projection matrix that got inserted
                                      Order is accoriding to order in CLASS_IDX_MAPPER
    """

    if isinstance(input_batch, np.ndarray): input_batch = torch.tensor(input_batch)

    class_idx_mapper = CLASS_IDX_MAPPER if not toy else {'class1': 0, 'class2': 1}

    input_batch = input_batch.to(device)
    
    # init flipper
    flipper = Flipper(perturbation_size=16, device=device)
    def forward_func(input_batch): return model(input_batch)

    """
    This loop loops over all classes to extract the according projection matrix in each iteration and then:
        1. generates subspace heatmaps of of all classes propagated through Q of one specific genre
        2. performs concept patch flipping of this batch
    """

    samples_per_class = int(input_batch.size(0) / len(class_idx_mapper))

    all = []

    for layer_idx in [1, 4, 7, 10, 13]:

        aupcs = []
        aupcs_per_class = []
        for i, subspace_genre in tqdm(enumerate(class_idx_mapper.keys())):

            # build the composite with the projection matrix Q for genre at layer_idx
            #composite = get_class_composite(name_map, genre=genre_Q, layer_idx=layer_idx, path_to_Qs=path_to_Qs, device=device, standard_r=standard_r, n_c=num_concepts) ############## change standard r
            # load projection matrix
            U = load_projection_matrix(subspace_genre, layer_idx, path_to_U, device=device)

            subspace_heatmaps = []

            # loop that aggregates subspace heatmaps of each genre
            for j, genre_to_attribute in enumerate(class_idx_mapper.keys()):

                # pop-popQ, metal-popQ, disco-popQ, ...

                # extract samples of each class in each iteration
                class_batch = input_batch[i*samples_per_class:(i+1)*samples_per_class].clone().detach().to(device)

                # init heatmap generator
                # instanciate heatmap generator
                generator = HeatmapGenerator(model, U, name_map, sample_class=genre_to_attribute, num_concepts=num_concepts, layer_idx=layer_idx, case=case, device=device)
                class_subspace_heatmaps = generator.generate_subspace_heatmaps(class_batch, concept_flipping=True)

                subspace_heatmaps.append(class_subspace_heatmaps)

                del generator
                del class_subspace_heatmaps
                torch.mps.empty_cache()

            # flip whole batch (all classes)
            aupc_scores_per_instance, _, _ = flipper(forward_func, input_batch, torch.concat(subspace_heatmaps, dim=0))
            class_aupcs = aupc_scores_per_instance.mean(axis=-1)

            # class_aupcs is of shape [num_classes,]
            aupcs.append(class_aupcs)


        # [num_classes, num_classes]
        aupcs_per_class = np.stack(aupcs, axis=0)

        all.append(aupcs_per_class)

        # TODO: dont need delta for intra class comparison. Only if we want to perform interclass comparison. 
        # Maybe later first try like this and postprocessing in notebook.

    return all


def load_projection_matrix(genre, layer_idx, path, device=torch.device('cpu')):
                
    # get best of 3 runs
    _, _, _, path_to_best_run, _ = get_best_run(os.path.join(path, f'{genre}/layer{layer_idx}'))

    with open(os.path.join(path_to_best_run, 'projection_matrix.pkl'), 'rb') as file:
            U = pickle.load(file)

    return torch.tensor(U, device=device)


def cf_random_subspace(model, input_batch, name_map, layer_idx, dim, case=None, device=torch.device('cpu'), permutations=3, num_concepts=4):

    if isinstance(input_batch, np.ndarray): input_batch = torch.tensor(input_batch)

    input_batch = input_batch.to(device)

    # init flipper
    flipper = Flipper(perturbation_size=16, device=device)
    def forward_func(input_batch): return model(input_batch)

    class_idx_mapper = CLASS_IDX_MAPPER if case != 'toy' else CLASS_IDX_MAPPER_TOY

    # get subspace heatmaps for every sample in batch
    n_classes = len(class_idx_mapper.keys())
    samples_per_class = input_batch.size(0) // n_classes

    # sample random projection matrix Q
    U = ortho_group.rvs(dim)

    for i in tqdm(range(permutations)):

        # permute Q for each iteration
        mask = np.random.permutation(dim)
        U = torch.tensor(U[:, mask], dtype=input_batch.dtype, device=device)

        subspace_heatmaps = []
        for i, genre in enumerate(class_idx_mapper.keys()):

            # extract samples of each class in each iteration
            class_batch = input_batch[i*samples_per_class:(i+1)*samples_per_class].clone().detach()

            # instanciate heatmap generator
            generator = HeatmapGenerator(model, U, name_map, sample_class=genre, num_concepts=num_concepts, 
                                         layer_idx=layer_idx, case=case, device=device)
            
            class_subspace_heatmaps = generator.generate_subspace_heatmaps(class_batch, concept_flipping=True)
            subspace_heatmaps.append(class_subspace_heatmaps)

        subspace_heatmaps = torch.cat(subspace_heatmaps, dim=0)
    subspace_heatmaps = np.array(subspace_heatmaps.cpu())

    return subspace_heatmaps


def sample_random_Q(dim, device=torch.device('cpu')):
    Q = ortho_group.rvs(dim)
    return torch.tensor(Q, device=device)


def perform_cf(model, input_batch, name_map, out, path=None, layer_idcs=[1,4,7,10,13], 
               num_concepts=[2,4,8,16], toy=False, prefix='', device=torch.device('cpu')):
    r"""
    This function calculates the aupcs per class at different layers for different number of subspaces

    saves evaluation data to pickle files. pickle files are named after the configuration (concepts, layer_idx)
    and contain aupc scores per instance for each class.

    -> AUPC array has shape [num_classes, samples_per_class]
    """

    dims = [32, 32, 64, 64, 128] if not toy else [8, 8, 16, 16, 16]

    for k in num_concepts:

        for i, layer_idx in enumerate(layer_idcs):

            print(f'Performing concept patch flipping for {k} subspaces at layer {layer_idx}')

            if prefix == 'random':
                aupc_scores_per_instance, _, _, _ = cf_random_subspace(model, input_batch, name_map, layer_idx, 
                                                                    dim=dims[i], device=device, permutations=3, num_concepts=k)            
            else:    
                aupc_scores_per_instance, _, _, _ = \
                    concept_flipping(model, input_batch, name_map, layer_idx, os.path.join(path, f'{k}_concepts'), num_concepts=k, device=device)
            
            conf_out = os.path.join(out, f'{prefix}/{k}_concepts')
            if not os.path.exists(conf_out): os.makedirs(conf_out)

            with open(os.path.join(conf_out, f'aupcs_layer_{layer_idx}.pkl'),'wb') as f:
                pickle.dump(np.stack(aupc_scores_per_instance, axis=0), f)



def sep_and_peak(model, input_batch, name_map, out, path=None, layer_idcs=[1,4,7,10,13], 
               num_concepts=[2,4,8,16], toy=False, prefix='', device=torch.device('cpu')):
    r"""
    Compute seperability and peakness.
    NOTE: under development
    """

    dims = [32, 32, 64, 64, 128] if not toy else [8, 8, 16, 16, 16]

    all = []

    for k in num_concepts:
        sep = []
        seperr = []
        peak = []
        peakerr = []

        for i, layer_idx in enumerate(layer_idcs):

            print(f'Performing concept patch flipping for {k} subspaces at layer {layer_idx}')

            if prefix == 'random':
                RU = cf_random_subspace(model, input_batch, name_map, layer_idx, 
                                                                    dim=dims[i], device=device, permutations=3, num_concepts=k)            
            else:    
                RU = \
                    concept_flipping(model, input_batch, name_map, layer_idx, os.path.join(path, prefix, f'{k}_concepts'), num_concepts=k, device=device)
                

            # b, cons, x, y
            frob_score = frob(RU, num_concepts)
                
            seperability_scores = (np.max(RU,1).sum((-2,-1)) - np.max(RU.sum((-2,-1)), 1)).squeeze()
            seperability = seperability_scores.mean()
            sep_stnd_err = seperability / np.sqrt(seperability_scores.shape[0])
            
            peakness_scores = np.max(RU, (-2,-1)).sum(1).squeeze()
            peakness = peakness_scores.mean()
            peak_stnd_err = peakness / np.sqrt(peakness_scores.shape[0])

            sep.append(seperability)
            seperr.append(sep_stnd_err)
            peak.append(peakness)
            peakerr.append(peak_stnd_err)
        
        all.append(np.stack((sep, seperr, peak, peakerr), axis=0))
    
    final = np.stack(all, axis=0)
            
    conf_out = os.path.join(out, f'{prefix}')
    if not os.path.exists(conf_out): os.makedirs(conf_out)

    with open(os.path.join(conf_out, f'sep_and_peak.pkl'),'wb') as f:
        pickle.dump(np.stack(final, axis=0), f)

    return final 


def frob(RU, num_concepts):

    # Step 1: Expand dimensions to compute pairwise differences
    # data[:, None, :, :, :] has shape (batch, 1, num_m, i, j)
    # data[:, :, None, :, :] has shape (batch, num_m, 1, i, j)
    diff = RU[:, None, :, :, :] - RU[:, :, None, :, :]

    # Step 2: Square the differences
    squared_diff = diff**2

    # Step 3: Sum over the last two dimensions (i, j) to get squared Frobenius norms
    squared_fro_norms = np.sum(squared_diff, axis=(-2, -1))

    # Step 4: Take the square root to obtain Frobenius norms
    fro_norms = np.sqrt(squared_fro_norms)

    # Step 5: Sum over all pairs (k, l) for k < l
    # We mask out unwanted pairs (self-pairs and upper triangle)
    mask = np.triu(np.ones((num_concepts, num_concepts), dtype=bool), k=1)
    total_fro_norms = np.sum(fro_norms[:, mask], axis=-1)
    
    combinations = num_concepts*(num_concepts-1)/2

    return total_fro_norms.mean() / combinations


            
# use for cuda
def main():

    random.seed(42)

    device = torch.device('cuda')
    #device = torch.device('mps')

    #path_to_data = '/Users/samuelharck/Desktop/masterthesis/other_datasets/GTZAN16k'
    path_to_data = '/input-data'

    # Load model
    path_to_model = os.path.join('/home/sharck/models/3s_ts/dr0.4_lr0.0004_bs16_wd0.0001/fold1',  'best_model_2000.pth')
    conf = ((32,32,64,64,128), 128, ((2,2), (2,2), (2,2), (2,2), (2,2)))
    model = VGGType(n_filters=conf[0], n_dense=conf[1], pool_kernels=conf[2], dropout=0.4, 
                    input_size=(128,128), conv_bn=False, dense_bn=False, block_depth=1)
    checkpoint = torch.load(path_to_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 

    # init composite for LRP
    gamma = 0.4
    stab1 = 1e-7
    stab2 = 1e-7
    eps = 1e-7

    name_map = [
        # block 1
        (['features.0'], WSquare(stabilizer=stab1)),
        # block 2
        (['features.3'], Gamma(gamma=gamma, stabilizer=stab2)),
        # block 3
        (['features.6'], Gamma(gamma=gamma, stabilizer=stab2)),
        # last conv block
        (['features.9'], Gamma(gamma=gamma/2, stabilizer=stab2)),

        (['features.12'], Gamma(gamma=gamma/4, stabilizer=stab2)),
        # fc block
        (['classifier.0'], Epsilon(epsilon=eps)),
        (['classifier.3'], Epsilon(epsilon=eps)),
        (['classifier.6'], Epsilon(epsilon=eps)),
    ]

    # get input batch (whole validation set with 3 chunks per soundtrack) = 600 samples
    input_batch, _ = get_data_main(path_to_data, fold=1, num_folds=5, samples_per_class=20, seed=42, num_chunks=3)

    for opt in np.arange(0):

        for alg in ['random', 'dsa', 'drsa']:

            #for locs in [25]:            
                
            #root = '/Users/samuelharck/Desktop/subs/rand'
            #out = f'/Users/samuelharck/Desktop/ev_test'
            root = f'/home/sharck/drsa/models/last_drsa/{opt}'
            out = f'/home/sharck/drsa/eval600/{opt}'
            path_to_subs = os.path.join(root, alg)

            _ = sep_and_peak(model, input_batch, name_map, out=out, path=path_to_subs, layer_idcs=[1,4,7,10,13], 
                            num_concepts=[2,4,8,16], prefix=alg, device=device)

    out = '/home/sharck/drsa/eval600'

    # standard R
    # instanciate flipper
    if isinstance(input_batch, np.ndarray): input_batch = torch.tensor(input_batch)
    pf = PixelFlipping(model, input_batch, perturbation_size=16, perturbation_mode='constant', num_classes=10, device=device)

    aupc_scores, _, _, _ = pf(configuration_grid=[{'convolutional': ('gamma', 0.4),  'dense': ('epsilon', 1e-7), 'first_layer': ('wsquare',)},], plot=False, scaled_gamma='peak4')

    aupc_scores = aupc_scores['gamma_0.4_epsilon_1e-07_wsquare']

    if not os.path.exists(out): os.makedirs(out)
    with open(os.path.join(out, 'standard_R.pkl'),'wb') as f:
        pickle.dump(aupc_scores, f)


if __name__ == '__main__':
    main()
