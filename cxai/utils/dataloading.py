import os
import torch
import torchaudio
import librosa

from cxai.utils.sound_processing import get_slice, peak_normalizer
from cxai.utils.constants import CLASS_IDX_MAPPER


class Loader():
    """
    This class serves as dataloader for specific cases like evaluation procedures or data preparation methodologies.
    """

    def __init__(self, case='gtzan'):

        self.sample_rate = 16000
        self.case = case

        if self.case == 'gtzan':
            n_fft = 800
            hop_length = 360
            self.n_mels = 128
            self.slice_length = 3
            self.width = 128
        else:
            n_fft = 480
            hop_length = 240
            self.n_mels = 64
            self.width = 64

        self.wav2spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.spec2mel = torchaudio.transforms.MelScale(n_mels=self.n_mels, n_stft=n_fft // 2 + 1, sample_rate=self.sample_rate)
        
    
    def load(self, path_to_audio,  num_chunks=1, startpoint=0, return_wav=False):
        """
        Loads sample, slices and normalizes it by peak and converts it into a mel-spectrogram.
        """
        wav, _ = torchaudio.load(path_to_audio)
        wav = wav.requires_grad_(False)

        if self.case == 'gtzan':
            wav = get_slice(wav, slice_length=self.slice_length, num_chunks=num_chunks,
                            start_point=startpoint, sample_rate=self.sample_rate)
        
        wav = peak_normalizer(wav)
        mel_normed = self.transform_wav(wav)

        if return_wav:
            return wav, mel_normed

        return mel_normed
    

    def load_batch_toy(self, songlist):
        samples = []
        for name in songlist:
            samples.append(self.load(name))
        return torch.stack(samples, dim=0).view(-1, 1, self.n_mels, self.width)
    

    def load_batch(self, songlist, startpoints):
        samples = []
        for name, startpoint in zip(songlist, startpoints):
            samples.append(self.load(name, startpoint=startpoint))
        return torch.stack(samples, dim=0).view(-1, 1, self.n_mels, self.width)
        
    

    def transform_wav(self, wav, return_all=False, clamp=True):
        """
        Transforms wav-form audio into logmel-spectrogram.
        """
        # complex spectrogram
        spec = self.wav2spec(wav)
        # mel spectrogram
        mel = self.spec2mel(torch.abs(spec))
        # convert mel to log10-scale
        logmel = torch.log10(mel + 1e-7)
        # clamp outliers
        if clamp: logmel = torch.clamp(logmel, -4)

        if return_all:
            # for Mel2Audio
            mag, phase = librosa.magphase(spec.numpy())
            return wav.numpy(), mag[..., :self.width], phase[..., :self.width], mel.numpy()[..., :self.width]
        
        # standard return
        if self.case == 'gtzan':
            return logmel[..., :self.width].reshape(-1,1,self.n_mels,self.width)
        else:
            return logmel[..., 1:-2].reshape(-1,1,self.n_mels,self.width)


    def load_all_representations(self, path_to_audio, startpoint=None):
        r"""
        Function transforms a wav-form signal into a mel-spectrogram  and returns all intermedieate representations.
        These include: original wav, magnitude spectrogram, phase spectrogram, mel spectrogram
        """
        return self.load(path_to_audio, startpoint, return_all=True)
    


###################################### functions to load paths to songs ######################################


def get_songlist(path, 
                 genre: str = None,
                 excluded_folds: list = None,
                 num_folds: int = 5, 
                 return_list: bool = True, 
                 genres: dict = CLASS_IDX_MAPPER):
    """
    Function to get list of audios of specific genre in defined folds.
    -----
    Args:
        genre           (str): genre to load instances from
        excluded_folds  (list): list of folds that should be EXCLUDED. 
        num_folds       (int): total number of folds
        return_list     (bool): return flag if list or dict should be returned by this function
        genre_dict      (dict): dict which maps genre strings to int
    Returns: 
        songlist
    """

    assert not path.startswith('.'), 'path has to be an absolute path'

    genres = [genre] if genre else genres
    songpaths = [] if return_list else {}

    for key in genres:
        songs_of_genre = get_songs_of_genre(path, key, excluded_folds, num_folds)
        if return_list:
            songpaths.extend(songs_of_genre)
        else:
            songpaths[key] = songs_of_genre
    return songpaths


def get_songs_of_genre(path, genre, excluded_folds: list = None, num_folds: int = 10) -> list:
    """
    Returns list of absolute paths to all samples of the specified genre
    """
    
    # case train and cross-validation: concatenate all folds together exept of the validation fold
    paths_to_songs = []

    for fold in range(1,num_folds+1):
        if excluded_folds is not None and fold in excluded_folds:
            continue

        list_filename = os.path.join(path, f'{num_folds}folds/fold_' + str(fold) + '.txt')
        with open(list_filename) as f:
            lines = f.readlines()
        # extend songlist
        for line in lines:
            line = line.strip()
            if str.split(line, '/')[0] == genre:
                path_to_sample = os.path.join(path, 'genres_original', line)
                paths_to_songs.append(path_to_sample)
    
    return paths_to_songs


def get_toy_samplelist(path, toyclass=None, splits: str=None) -> list:
    """
    Loads the paths to every sample of toy data
    """

    splits = ['train', 'valid', 'test'] if splits is None else [splits]

    samplelist = []
    # get all songs from toy data
    for split in splits:
        list_filename = os.path.join(path, split + '_split.txt')
        with open(list_filename) as f:
            lines = f.readlines()

        for line in lines:
            if toyclass:
                if str.split(line, '/')[0] == toyclass:
                    samplelist.append(os.path.join(path, line.strip()))
            else:
                samplelist.append(os.path.join(path, line.strip()))
    
    return samplelist


def get_songlist_random(path, num_folds=5) -> list:

    songlist = []
    for fold in range(1,num_folds+1):
        list_filename = os.path.join(path, f'fold_{fold}.txt')
        with open(list_filename) as f:
            lines = f.readlines()
        # extend songlist
        songlist.extend([line.strip() for line in lines])

    return songlist


def shuffle_and_truncate_databatch(data_batch, paths_to_songs, N, seed=42):

    # instantiate local generator
    local_gen = torch.Generator()
    local_gen = local_gen.manual_seed(seed)
    perm_mask = torch.randperm(data_batch.size(0), generator=local_gen)

    data_batch = data_batch[perm_mask][:N]
    paths_to_songs = [paths_to_songs[i] for i in perm_mask[:N]]

    return data_batch, paths_to_songs


"""def get_song_list(path_to_txt, split):
    
    if split.startswith('fold'):
        filename = '%s.txt' % split
    elif split.endswith('filtered'):
        filename = '%s.txt' % split
    else:
        filename = '%s_split.txt' % split

    list_filename = os.path.join(path_to_txt, filename)
    with open(list_filename) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]"""