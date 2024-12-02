import os
from typing import Tuple, List

import numpy as np
import torch
import torchaudio
import librosa

from cxai.utils.sound import get_slice, peak_normalizer
from cxai.utils.constants import CLASS_IDX_MAPPER, AUDIO_PARAMS


class Loader():
    """
    This class serves as dataloader for specific cases like evaluation procedures or data preparation processes.

    Attributes:
        TODO
    """

    def __init__(
        self,
        case: str | None = None,
        sample_rate: int = 16000,
        n_fft: int = 800,
        hop_length: int = 360,
        n_mels: int = 128,
        slice_length: int = 3,
        width: int = 128
    ) -> None:
        """Init audio loader class.

        Args:
            case (str): Options ['gtzan', 'toy'].
            sample_rate (int): Sample rate.
            n_fft (int): Nuber of fft bins for STFT.
            hop_length (int): Hop size between bins.
            n_mels (int): Number of mel bins.
            slice_length (int): Time length of audio snippets.
            width (int): Width of mel spectrogram.
        """
        # load audio params
        if case is not None and case in list(AUDIO_PARAMS.keys()):
            self.sample_rate = AUDIO_PARAMS[case]['sample_rate']
            n_fft = AUDIO_PARAMS[case]['n_fft']
            hop_length = AUDIO_PARAMS[case]['hop_length']
            self.n_mels = AUDIO_PARAMS[case]['n_mels']
            self.width = AUDIO_PARAMS[case]['mel_width']
            self.slice_length = AUDIO_PARAMS[case].get('slice_length', 0)
        else:
            self.sample_rate = sample_rate
            self.n_mels = n_mels
            self.slice_length = slice_length
            self.width = width
        # init waveform to audio converter
        self.wav2spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, 
            hop_length=hop_length, 
            power=None
        )
        # init freqeuency to mel scale converter
        self.spec2mel = torchaudio.transforms.MelScale(
            n_mels=self.n_mels, 
            n_stft=n_fft // 2 + 1, 
            sample_rate=self.sample_rate
        )
        
    def load(
        self, 
        path_to_audio: str,  
        num_chunks: int = 1, 
        startpoint: int = 0, 
        return_wav: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        
        """Loads audio sample, slices and normalizes it by peak and converts it 
        into a log-mel-spectrogram.

        Args:
            path_to_audio (str): Path to the audio to load.
            num_chunks (int): Number of slices that should be extracted from the audio.
            startpoint (int): Startpoint in seconds where to extract the first chunk.
            return_wav (bool): Control flag if the waveform audio should also be returned.

        Returns:
            mel_normed (torch.Tensor): Log-mel-spectrogram.
        """
        wav, _ = torchaudio.load(path_to_audio)
        wav = wav.requires_grad_(False)

        # slice the audio
        if self.slice_length != 0:
            wav = get_slice(
                wav, 
                self.slice_length,
                startpoint,
                num_chunks,
                self.sample_rate
            )
        # normalize waveform by peak
        wav = peak_normalizer(wav)
        # transform to mel spectrogram
        mel_normed = self.transform_wav(wav)

        if return_wav:
            return wav, mel_normed
        return mel_normed

    def load_batch(self, songlist: list, startpoints: list = None) -> torch.Tensor:
        """Loads snippets of a given batch.
        
        NOTE: for now this function only loads one snippet per smaple. 
        Will be changed to load more snippets.
        """
        if startpoints == None: startpoints = np.zeros(len(songlist))
        samples = []
        for name, startpoint in zip(songlist, startpoints):
            samples.append(self.load(name, startpoint=startpoint))
        return torch.stack(samples, dim=0).view(-1, 1, self.n_mels, self.width)

    def transform_wav(
        self, 
        wav: torch.Tensor, 
        return_all: bool = False, 
        clamp: bool = True
    ) -> torch.Tensor:
        """Transforms wav-form audio into logmel-spectrogram.
        
        Args:
            wav (torch.Tensor): Waveform audio.
            return_all (bool): Flag to control the returns. If True, all audio representations are returned. 
            clamp (bool): Clamp the log-amplitudes at a negative value.

        Returns:
            logmel (torch.Tensor): log-mel-spectrogram.
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
            # NOTE: when return_all==True, np.ndarrays are retuned
            mag, phase = librosa.magphase(spec.numpy())
            return wav.numpy(), mag[..., :self.width], phase[..., :self.width], mel.numpy()[..., :self.width]
        
        # standard return
        logmel = logmel[..., 1:self.width+1]
        assert logmel.shape[-1]==self.width, f"width of logmel-spectrogram ({logmel.shape[-1]}) \
            has to equal self.width ({self.width})."
        return logmel.reshape(-1,1,self.n_mels,self.width)

    def load_all_representations(self, path_to_audio: str, startpoint: int = None):
        """Calls self.load with return_all=True."""
        return self.load(path_to_audio, startpoint, return_all=True)
    

def shuffle_and_truncate_databatch(data_batch, paths_to_songs, N, seed=42):
    """Shuffle a readily loaded databatch."""

    # instantiate local generator
    local_gen = torch.Generator()
    local_gen = local_gen.manual_seed(seed)
    perm_mask = torch.randperm(data_batch.size(0), generator=local_gen)
    data_batch = data_batch[perm_mask][:N]
    paths_to_songs = [paths_to_songs[i] for i in perm_mask[:N]]

    return data_batch, paths_to_songs


def get_songlist(
    path, 
    genre: str = None,
    excluded_folds: list = None,
    num_folds: int = 5, 
    return_list: bool = True, 
    genres: dict = CLASS_IDX_MAPPER
) -> (List[str]):
    """Function to get list of audios of specific genre in defined folds.
    
    Args:
        genre (str): genre to load instances from
        excluded_folds (list): list of folds that should be EXCLUDED. 
        num_folds (int): total number of folds
        return_list (bool): return flag if list or dict should be returned by this function
        genre_dict (dict): dict which maps genre strings to int

    Returns: 
        songpaths (List[str]): Paths to songs.
    """

    #assert path.startswith('.'), 'path has to be an absolute path'
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
    """Returns list of absolute paths to all samples of the specified genre"""
    
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
    """Loads the paths to every sample of toy data"""

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