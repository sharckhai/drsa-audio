import os
import numpy as np
import torch
import random
from tqdm import tqdm

from cxai.utils.dataloading import Loader, get_songlist
from cxai.utils.constants import CLASS_IDX_MAPPER


def get_data_main(datapath, 
                    samples_per_class, 
                    fold: str = None,
                    genre: str = None, 
                    exclude_folds: list = None,
                    num_chunks: int = 1,
                    num_folds: int = 5,
                    N=None,
                    device=torch.device('cpu'), 
                    seed=42) -> np.ndarray:
    """
    This function loads randomly samples chunks from audios for pixelflipping evaluation.
    -----
    Args:
        datapath            (str): path to data
        samples_per_class   (int): how many instances per class should be loaded
        fold                (str): defines the fold which was used for validation (which fold should be excluded from dataloading)
        num_chunks          (int): number of chunks that should be extracted from the original signal
        num_folds           (int): total number of folds
    Returns:
        data_batch_tensor   (torch.Tensor): data batch
        loaded_samples      (list): list of sample names that were loaded
    """

    # instanciate data_batch tensor
    data_batch = []

    # define which folds to exclude from loading
    exclude_folds = np.delete(np.arange(1, num_folds+1), fold) if fold else exclude_folds
    sample_dict = get_songlist(datapath, genre, exclude_folds, num_folds, return_list=False)

    # create local random number generator to not affect the system wide seed
    local_random = random.Random()
    local_random.seed(seed)

    # init sample loader
    loader = Loader(case='gtzan')

    loaded_samples = []
    tqdm_desc = f'Loading {samples_per_class} data samples per class' if not genre else f'Loading {samples_per_class} data samples of genre {genre}'

    for genre_class in tqdm(sample_dict, total=len(sample_dict), desc=tqdm_desc, unit_scale=samples_per_class*num_chunks):

        samplelist = sample_dict[genre_class]
        local_random.shuffle(samplelist)

        assert samples_per_class <= len(samplelist), f'samples_per_class has to be smaller or equal to num samples in dataset class {genre_class}!'

        for i in range(samples_per_class):

            # load wavefrom and transform to mel speectrograms
            #mels = loader.load(os.path.join(datapath, 'genres_original', samplelist[i]), num_chunks=num_chunks, startpoint=local_random.randint(5,25) if num_chunks==1 else None)
            mels = loader.load(os.path.join(datapath, 'genres_original', samplelist[i]), num_chunks=num_chunks)
            mels = mels.requires_grad_(False).to(device)

            data_batch.extend(mels.cpu().numpy())
            loaded_samples.append(samplelist[i])
    
    data_batch_tensor = np.stack(data_batch, axis=0)

    return data_batch_tensor, loaded_samples


def get_data_toy(datapath, 
                split, 
                samples_per_class, 
                class_idx_mapper, 
                single_genre=None, 
                device=torch.device('cpu'), 
                seed=42) -> np.ndarray:
    """
    Loads a batch of data.
    -----
    Args:
        datapath (str): path to data
        split (str): 
        class_idx_mapper (dict): Dict that maps labels to class strings. Has to be in the form of: {0: 'class_name', 1: 'class2_name', ...}
    """

    # get sample names in test split
    samples = get_song_list(datapath, split)

    # create local random number generator to not affect the system wide seed
    local_random = random.Random()
    local_random.seed(seed)
    local_random.shuffle(samples)

    # init sample loader
    loader = Loader(case='toy')
    data_batch = []
    loaded_samples = []

    assert samples_per_class <= len(samples), 'samples_per_class has to be smaller or even than number of samples per class in test split!'

    # iterate over all classes
    for sample_class in tqdm(class_idx_mapper, total=len(class_idx_mapper), desc=f'Loading {samples_per_class} data samples per class', unit_scale=samples_per_class):
        
        # if a single genre is defined we only extract data samples from this genre
        if single_genre:
            if sample_class != single_genre:
                continue

        # sample counter and increase until num_samples_per_class is reached then break
        sample_counter = 0

        for sample in samples:
            if sample.startswith(sample_class):
                # load wavefrom and transform to mel speectrograms
                mels = loader.load(os.path.join(datapath, sample))
                mels = mels.requires_grad_(False).to(device)
                data_batch.extend(mels.cpu().numpy())
                loaded_samples.append(sample)
                sample_counter += 1

                if sample_counter == samples_per_class:
                    break

    data_batch_tensor = np.stack(data_batch, axis=0)
    return data_batch_tensor, loaded_samples


def get_song_list(path_to_txt, split):
    
    if split.startswith('fold'):
        filename = '%s.txt' % split
    elif split.endswith('filtered'):
        filename = '%s.txt' % split
    else:
        filename = '%s_split.txt' % split

    list_filename = os.path.join(path_to_txt, filename)
    with open(list_filename) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]