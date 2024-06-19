import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ortho_group
import pickle

import zennit
from zennit.rules import Epsilon, ZPlus, Norm, Pass, WSquare, Gamma, Flat, AlphaBeta, ZBox
from zennit.composites import SpecialFirstLayerMapComposite, NameMapComposite, NameLayerMapComposite
from zennit.canonizers import SequentialMergeBatchNorm, NamedMergeBatchNorm, CompositeCanonizer
from zennit.types import Linear, Convolution, Activation, BatchNorm

sys.path.append('../')

from create_model import VGGType
#from xai.pixelflipping.prep import get_data
#from drsa import drsa
from preprocessing import preprocess_data, get_songs, normalize_vectors
#from constants import CLASS_IDX_MAPPER




def save_data(activation_vectors, context_vectors, layer=None, sample_class=None, case='gtzan', model='bn', output_path=None):

    assert type(layer) == int, 'layer has to be defined and of type int'
    assert type(sample_class) == str, 'sample_class has to be defined and of type str'
    assert output_path is not None, 'please provide an output path to save the data'

    paired_dataset = list(zip(activation_vectors, context_vectors))

    # Specify the filename
    filename = f'dataset_layer{layer}.pkl'
    path = os.path.join(output_path, f'{case}/{model}/{sample_class}')
    filepath = os.path.join(path, filename)

    if not os.path.exists(path):
        os.makedirs(path)

    # Open a file in binary-write mode and save the dataset using pickle
    with open(filepath, 'wb') as file:
        pickle.dump(paired_dataset, file)


def load_and_normalize_data(filepath, device):
    
    with open(filepath, 'rb') as file:
        dataset = pickle.load(file)

    a, c = zip(*dataset)
    print(type(a))

    a, c = torch.tensor(np.array(a), device=device), torch.tensor(np.array(c), device=device)
    a, c = a.detach().requires_grad_(False), c.detach().requires_grad_(False)
    print(a.size()), print(c.size())
    
    return normalize_vectors(a), normalize_vectors(c)



def main():
    device = torch.device('cuda')

    path_to_data = '/input-data'
    output_path = '/home/sharck/drsa_datasets/'

    # Load model
    fold = 1
    model_path = os.path.join('/home/sharck/models/6s_gtzan/((64, 64, 100, 128, 128), 100)_BN/dr0.3_lr0.0001_bs16_wd0.0001_mm0.99',  'best_model_1700.pth')
    conf = ((64,64,100,128,128), 100, ((2,4), (2,2), (2,2), (2,2), (2,2)))
    model = VGGType(n_filters=conf[0], n_dense=conf[1], pool_kernels=conf[2], dropout=0.3, input_size=(128,256), conv_bn=True, dense_bn=True)
    # Load states from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    #params for lrp
    canonizer = SequentialMergeBatchNorm()
    gamma = 0.3
    stab1 = 1e-7
    stab2 = 1e-7
    stab3 = 1e-7

    name_map = [
        # block 1
        (['features.0'], WSquare(stabilizer=stab1)),
        (['features.3'], Gamma(gamma=gamma, stabilizer=stab2)),
        # block 2
        (['features.7'], Gamma(gamma=gamma, stabilizer=stab2)),
        (['features.10'], Gamma(gamma=gamma, stabilizer=stab2)),
        # block 3
        (['features.14'], Gamma(gamma=gamma/2, stabilizer=stab2)),
        (['features.17'], Gamma(gamma=gamma/2, stabilizer=stab2)),
        # last conv block
        (['features.21'], Gamma(gamma=gamma/2, stabilizer=stab2)),
        (['features.24'], Gamma(gamma=gamma/2, stabilizer=stab2)),

        (['features.28'], Gamma(gamma=gamma/4, stabilizer=stab2)),
        (['features.31'], Gamma(gamma=gamma/4, stabilizer=stab2)),
        # fc block
        (['classifier.0'], Epsilon(epsilon=stab3)),
        (['classifier.4'], Epsilon(epsilon=stab3)),
        (['classifier.8'], Epsilon(epsilon=stab3)),#, zero_params='bias')),
        #(['classifier.9'], MTMRule(stabilizer=1e-7))
    ]


    composite_name_map = NameMapComposite(
        name_map=name_map,
        canonizers=[canonizer],
    )

    class_idx_mapper = {"pop": 0, "metal": 1, "disco": 2, "blues": 3, "reggae": 4, "classical": 5, "rock": 6, "hiphop": 7, "country": 8, "jazz": 9}

    for genre in list(class_idx_mapper.keys()):
        for layer_idx in [19, 26, 33]:

            print(f'Creating dataset for genre: {genre} at layer: {layer_idx}')

            data_batch, loaded_samples = get_songs(path_to_data, sample_class=genre, excluded_folds=[fold], device=device)
            #data_batch = data_batch.detach().requires_grad_(False)

            # define layer where to extract activation and context vectors
            #layer = model.features[layer_idx]

            # extract activation and context vectors
            activation_vectors, context_vectors = preprocess_data(model, data_batch, composite_name_map, layer_idx, device=device, \
                                                                  class_idx=class_idx_mapper[genre], num_locations=20, case='gtzan')
            activation_vectors = activation_vectors.detach().cpu().numpy()
            context_vectors = context_vectors.detach().cpu().numpy()
            #print(activation_vectors.shape, context_vectors.shape)

            save_data(activation_vectors, context_vectors, layer=layer_idx, \
                      sample_class=genre, model='1700_big_norm_20', output_path=output_path)

            del activation_vectors
            del context_vectors



if __name__ == "__main__":
    main()