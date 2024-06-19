import os
import torch

import drsa
from drsa.cluster.getdrsadata import load_and_normalize_data


def main(args):
    device = torch.device('cuda')

    path_to_data = '/input-data'

    if args.conf == 1:
        path_to_models = '/home/sharck/drsa_models/big_norm_4_20locs'
        class_idx_mapper = {"pop": 0, "metal": 1, "disco": 2, "blues": 3, "reggae": 4, "classical": 5, "rock": 6, "hiphop": 7, "country": 8, "jazz": 9}

        for sample_class in class_idx_mapper.keys(): #, "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]:
            for layer_idx in [19, 26, 33]: #, 18]:
                activation_vectors, context_vectors = load_and_normalize_data(os.path.join(path_to_data, f'gtzan/1700_big_norm_20/{sample_class}/dataset_layer{layer_idx}.pkl'), \
                                                                            debice=device)
                # start optimization
                drsa.main(activation_vectors, context_vectors, os.path.join(path_to_models, sample_class, f'layer{layer_idx}'), 
                            num_concepts=4, steps=5000, runs=3, seed=42, device=device)
        
    elif args.conf == 2:
        path_to_models = '/home/sharck/drsa_models/vggish/4c'
        class_idx_mapper = {'class1': 0, 'class2': 1, 'class3': 2, 'class4': 3}

        for sample_class in class_idx_mapper.keys(): #, "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]:
            for layer_idx in [9, 14]:
                activation_vectors, context_vectors = load_and_normalize_data(os.path.join(path_to_data, f'{sample_class}/dataset_layer{layer_idx}.pkl'), \
                                                                            debice=device)
                # start optimization
                drsa.main(activation_vectors, context_vectors, os.path.join(path_to_models, sample_class, f'layer{layer_idx}'), 
                            num_concepts=2, steps=5000, runs=3, seed=42, device=device)
                
    else:
        path_to_models = '/home/sharck/drsa_models/vggish/2c'
        class_idx_mapper = {'class1': 0, 'class2': 1}

        for sample_class in class_idx_mapper.keys(): #, "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]:
            for layer_idx in [9, 14]:
                activation_vectors, context_vectors = load_and_normalize_data(os.path.join(path_to_data, f'{sample_class}/dataset_layer{layer_idx}.pkl'), \
                                                                            debice=device)
                # start optimization
                drsa.main(activation_vectors, context_vectors, os.path.join(path_to_models, sample_class, f'layer{layer_idx}'), 
                            num_concepts=2, steps=5000, runs=3, seed=42, device=device)
        

            



import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Ddifferent calls of training')
    parser.add_argument('--conf', type=int, required=False)
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Call main() with the parsed arguments
    main(args)
