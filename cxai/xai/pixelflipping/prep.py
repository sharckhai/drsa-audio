import os
import sys
import numpy as np
import torch
import random
from tqdm import tqdm

sys.path.append('../')

from cxai.utils.all import load_sample, HiddenPrints, Loader, get_songlist, shuffle_and_truncate_databatch
from cxai.utils.constants import CLASS_IDX_MAPPER



def get_data(model, 
             datapath, 
             split, 
             samples_per_class, 
             class_idx_mapper, 
             slice_length=6,
             num_chunks=4,
             case='128_256',
             normalize=True, 
             single_genre=None, 
             device=torch.device('cpu'), 
             seed=42) -> np.ndarray:
    """
    
    Parameters:
    ----------
    class_idx_mapper: dict
        Dict that maps labels to class strings. Has to be in the form of: {0: 'class_name', 1: 'class2_name', ...}

    """

    

    # get sample names in test split
    samples = get_song_list(datapath, split)
    #samples.sort()   ###################################################################### IMPORATNT use for other function GTZAN

    # create local random number generator to not affect the system wide seed
    local_random = random.Random()
    local_random.seed(seed)
    local_random.shuffle(samples)

    # init sample loader
    loader = Loader(case='toy')
    
    # instanciate data_batch tensor
    data_batch = []
    loaded_samples = []

    assert samples_per_class <= len(samples), 'samples_per_class has to be smaller or even than number of samples per class in test split!'

    #print('Loading data...')

    # iterate over all classes
    for sample_class in tqdm(class_idx_mapper, total=len(class_idx_mapper), desc=f'Loading {samples_per_class} data samples per class', unit_scale=samples_per_class):
        
        # if a single genre is defined we only extract data samples from this genre
        if single_genre:
            if sample_class != single_genre:
                continue

        # sample counter and increase until num_samples_per_class is reached then break
        sample_counter = 0
        #for i in range(samples_per_class):
        for sample in samples:
        #while sample_counter < samples_per_class:
            
            #sample = samples[i+(class_idx*samples_per_class)]

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


def get_data_new_new(
        datapath, 
        excluded_folds=None,
        genre=None, 
        samples_per_class=10, 
        sample_rate=16000,
        num_chunks: int = 1,
        device=torch.device('cpu'), 
        seed=42,
        ) -> np.ndarray:
    """
    
    Parameters:
    ----------
    class_idx_mapper: dict
        Dict that maps labels to class strings. Has to be in the form of: {0: 'class_name', 1: 'class2_name', ...}

    """

    # instanciate data_batch tensor
    data_batch = []

    # get sample names in test split
    songnames = get_songlist(datapath, excluded_folds=excluded_folds, return_dict=False)

    # create local random number generator to not affect the system wide seed
    local_random = random.Random()
    local_random.seed(seed)
    local_random.shuffle(songnames)
    
    loaded_samples = []

    assert samples_per_class <= len(songnames), 'samples_per_class has to be smaller or even than number of samples per class in test split!'

    if sample_rate == 16000:
        loader = Loader(sample_rate=sample_rate)
    else:
        loader = Loader(sample_rate=sample_rate)

    # iterate over all classes
    for genre_key in tqdm(CLASS_IDX_MAPPER, total=len(CLASS_IDX_MAPPER), desc=f'Loading {samples_per_class} correctly classified data samples per class'):
        
        # if a single genre is defined we only extract data samples from this genre
        if genre:
            if genre_key != genre:
                continue

        # sample counter and increase until num_samples_per_class is reached then break
        sample_counter = 0
        for path_to_sample in songnames:
            if path_to_sample.split('/')[-2] == genre_key:

                # load wavefrom and transform to mel speectrograms
                mels = loader.load(path_to_sample, num_chunks=num_chunks)
                mels = mels.requires_grad_(False).to(device)

                data_batch.extend(mels.cpu().numpy())
                loaded_samples.append(path_to_sample)
                sample_counter += 1

                if sample_counter == samples_per_class:
                    break
    
    data_batch_tensor = np.stack(data_batch, axis=0)

    return data_batch_tensor, loaded_samples


def get_data_new_last(datapath, 
                    samples_per_class, 
                    fold: str = None,
                    genre: str = None, 
                    exclude_folds: list = None,
                    num_chunks: int = 1,
                    num_folds: int = 10,
                    N=None,
                    device=torch.device('cpu'), 
                    seed=42) -> np.ndarray:
    """
    
    Parameters:
    ----------
    class_idx_mapper: dict
        Dict that maps labels to class strings. Has to be in the form of: {0: 'class_name', 1: 'class2_name', ...}

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



def get_data_new(model, 
             datapath, 
             split, 
             samples_per_class, 
             class_idx_mapper, 
             num_chunks: int = 1,
             genre: str = None, 
             exclude_folds: list = None,
             num_folds: int = 10,
             device=torch.device('cpu'), 
             seed=42) -> np.ndarray:
    """
    
    Parameters:
    ----------
    class_idx_mapper: dict
        Dict that maps labels to class strings. Has to be in the form of: {0: 'class_name', 1: 'class2_name', ...}

    """

    # instanciate data_batch tensor
    data_batch = []

    # get sample names in test split

    # TODO: include genre here if genre is passed as argument and delete the genre filtering from main loop beneath in this fucntion

    #samples = get_song_list(datapath, split)
    
    # already only returns songs of 1 genre if specified
    samples = get_songlist(datapath, genre, exclude_folds, num_folds)

    # create local random number generator to not affect the system wide seed
    local_random = random.Random()
    local_random.seed(seed)
    local_random.shuffle(samples)
    
    assert samples_per_class <= len(samples), 'samples_per_class has to be smaller or even than number of samples per class in test split!'

    loader = Loader(sample_rate=16000)
    loaded_samples = []

    # iterate over all classes
    for class_idx, genre_class in tqdm(enumerate(class_idx_mapper), total=len(class_idx_mapper), \
                                       desc=f'Loading {samples_per_class} correctly classified data samples per class'):
        
        # if a single genre is defined we only extract data samples from this genre
        if genre:
            if genre_class != genre:
                continue

        # sample counter and increase until num_samples_per_class is reached then break
        sample_counter = 0
        for sample in samples:
            if sample.startswith(genre_class):

                # load wavefrom and transform to mel speectrograms
                mels = loader.load(os.path.join(datapath, 'genres_original', sample), num_chunks=num_chunks)
                mels = mels.requires_grad_(False).to(device)

                # test which samples get predicted right
                filtered_samples = evaluate_samples(model, mels, class_idx)

                if len(filtered_samples) > 0:
                    data_batch.extend(mels.cpu().numpy())
                    loaded_samples.append(sample)
                    sample_counter += 1

                if sample_counter == samples_per_class:
                    break
    
    data_batch_tensor = np.stack(data_batch, axis=0)

    return data_batch_tensor, loaded_samples



'''def get_mels_of_sample(datapath, sample, slice_length=6, num_chunks=4, case='128_256', normalize=True):
    """
    This functioo return a mel spectrogram extracted from the given audio sample. One can define the number of chunks in which the audio gets sliced. Default is 1 chunk.
    """

    # get mel sample, slice_length and start_point is not needed for case=toy but we can simply pass all args to load_sample(), case parameter accounts for this in load_sample()
    path_to_sample = os.path.join(datapath, sample) if case == 'toy' else os.path.join(datapath, 'genres_original', sample)

    mel, _ = load_sample(path_to_sample, slice_length=slice_length, start_point=10, num_chunks=num_chunks, to_mel=True, case=case, \
                         normalize=normalize, audio_normalization=None if normalize else 'peak')

    # if num_chunks = 1, add batch dimension
    mel = mel[None] if len(mel.size()) < 4 else mel

    return mel'''



def evaluate_samples(model, mels, target):

    assert len(mels.size()) == 4, 'wrong dimension of mel-spectrogram batch'

    pred = torch.argmax(model(mels), dim=-1)
    filterd_samples = mels[pred == target]

    return filterd_samples



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

