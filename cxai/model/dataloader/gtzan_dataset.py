import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio_augmentations import (RandomApply, Compose, Noise, 
                                      Gain, HighLowPass, PitchShift)

from cxai.utils.sound import round_down, peak_normalizer
from cxai.utils.constants import CLASS_IDX_MAPPER, AUDIO_PARAMS


class AudioDataset(Dataset):
    """Dataset class.

    Loads waveform audio, applies transforms and normalizes by peak.
    Transforms with stft into complex spectrogram an applies time stretch there.
    Transforms spectrpogram into log-mel-spectrogram by projecting mel frequencies 
    onto the log10 scale.

    Attributes:
        TODO
        data_path (str | Path): path to data folder root
        split (str): options ['train', 'validation']
        validation_fold (int): defines which fold is used for validation
        mask_param (int): maximum number of bins masked per dimension of the mel
        wav_transform (bool): flag to switch on/off waveform transformations
        mel_transform (bool): flag to switch on/off mel transformations
        device (str | torch.device): device
    """

    def __init__(
        self, 
        data_path: str | Path,
        split: str,
        validation_fold: int = 1,
        mask_param: int = 40,
        wav_augment: bool = True, 
        mel_augent: bool = True,
        device: str | torch.device = torch.device('mps')
    ) -> None:
        """Init dataset class for audio loading and augmentation.
        
        Args:
            data_path (str | Path): path to data folder root
            split (str): options ['train', 'validation']
            validation_fold (int): defines which fold is used for validation
            mask_param (int): maximum number of bins masked per dimension of the mel
            wav_augment (bool): flag to switch on/off waveform transformations
            mel_augent (bool): flag to switch on/off mel transformations
            device (str | torch.device): device
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # data params
        self.data_path = data_path if isinstance(data_path, Path) else Path(data_path)
        self.split = split
        self.genres = CLASS_IDX_MAPPER
        self.validation_fold = validation_fold

        # audio params
        self.sample_rate = AUDIO_PARAMS['gtzan']['']
        self.slice_length = AUDIO_PARAMS['gtzan']['slice_length']
        self.num_chunks = AUDIO_PARAMS['gtzan']['num_chunks']
        self.n_fft = AUDIO_PARAMS['gtzan']['n_fft']
        self.hop_length = AUDIO_PARAMS['gtzan']['hop_length']
        self.n_mels = AUDIO_PARAMS['gtzan']['n_mels']

        # infer further audio statistics
        self.window_size = self.sample_rate * self.slice_length
        self.slice_hop = int(round_down(((29 - self.slice_length) / (self.num_chunks-1)), 1) * self.sample_rate)
        # we want square spectrograms
        self.time_bins = self.n_mels
        # nomalization and augmentation params and flags
        self.mask_param_time = mask_param
        self.wav_augment = wav_augment
        self.mel_augent = mel_augent
        # get songlist and augemnattions
        self.songlist, self.labels = self.get_songs()
        # get audio transformations aund aufgemntations
        self.init_adio_transforms()
        
    def init_adio_transforms(self):
        """Initialize audio transforms and augmentations."""
        self.wav2spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            power=None
        )
        self.spec2mel = torchaudio.transforms.MelScale(
            sample_rate=self.sample_rate, 
            n_mels=self.n_mels, 
            n_stft=self.n_fft // 2 + 1
        )
        self.time_stretch = torchaudio.transforms.TimeStretch(
            hop_length=self.hop_length, 
            n_freq=self.n_fft // 2 + 1
        )
        # get waveform augmentations
        if self.wav_augment:
            self.wav_augmentation = self.wav_augmentations()
        
    def get_songs(self, validation_fold: int) -> list:
        """Function to accumulate all songpaths for the defined split (i.e., accumulates songs from multiple folds).

        Args:
            validation_fold (int): defines the validation fold. All other folds are combined for training.

        Returns:
            songlist (list): songpaths and labels
            labels (list): labels
        """
        assert validation_fold is not None, 'Pease provide validation fold idx.'

        songlist, labels, supfold = [], [], []
        # create big list (supfold) as combination of several folds
        for fold in self.data_path.glob('*.txt'):
            # train
            if self.split == 'train' and int(fold.name[-5]) != validation_fold:
                with open(fold) as f:
                    lines = f.readlines()
                supfold.extend(lines)
            # validation
            elif self.split != 'train' and int(fold.name[-5]) == validation_fold:
                with open(fold) as f:
                    lines = f.readlines()
                supfold.extend(lines)
                
        for line in supfold:
            path_to_song = line.strip()
            songlist.append(self.data_path / 'genres_original' / path_to_song)
            labels.append(self.genres[path_to_song.split('/')[0]])

        return songlist, labels

    def wav_augmentations(self):
        """Returns a compose of augekmentations on waveform audios.
        
        Returns:
            augementations (torchaudio_augmentations.Compose): waveform augmentations
        """
        transforms = [
            RandomApply([Gain(min_gain=-12, max_gain=3)], p=0.5),
            RandomApply([PitchShift(n_samples=self.window_size, 
                                    sample_rate=self.sample_rate, 
                                    pitch_shift_max=12, 
                                    pitch_shift_min=-12)], p=0.3),
            RandomApply([HighLowPass(sample_rate=self.sample_rate, 
                                    lowpass_freq_low = 1400,
                                    lowpass_freq_high = 4000,
                                    highpass_freq_low = 200,
                                    highpass_freq_high = 1400,)], p=0.4),
            RandomApply([Noise(min_snr=1e-3, max_snr=1e-1)], p=0.3),
        ]
        return Compose(transforms=transforms)
    
    def get_slice(self, wav: torch.Tensor, label: torch.Tensor):
        """Extract slice/slices from original waveform audio.
        
        Args:
            wav (troch.Tensor): waveform audio

        Returns:
            sliced_wavs (troch.Tensor): waveform audios
        """
        if self.split == 'train':
            # if train we sample a random part of the wav with length = slice_length
            random_index = random.randint(0, wav.size(1) - self.window_size - 1)
            sliced_wavs = wav[:, random_index : random_index + self.window_size]
        else:
            # unfold along the time dimension
            sliced_wavs = wav[:, :29*self.sample_rate].unfold(1, self.window_size, \
                                                      self.slice_hop).reshape(-1, 1, self.window_size)
            # we have to duplicate the label as often as we have created slices for validation
            label = label.repeat(self.num_chunks)
        return sliced_wavs, label

    def __getitem__(self, index):

        # songpath and label
        path_to_song = self.songlist[index]
        label = torch.Tensor([self.labels[index]]).requires_grad_(False)

        # load audio waveform
        wav, _ = torchaudio.load(path_to_song)
        wav = wav.requires_grad_(False)#.to(self.device)
        # slice audio
        wav, label = self.get_slice(wav, label)
        # perform volume-normalization of waveform by peak
        wav = peak_normalizer(wav)
        # data augmentation
        if self.wav_transform:
            wav = self.wav_augmentation(wav)

        # perform stft  
        spec = self.wav2spec(wav) # complex spectrogram
        # perform time stretching
        if self.mel_transform: 
                spec = self.time_stretch(spec, round(random.uniform(0.8,1.2), 3))
        # transform to mel spectrorgam
        mel = self.spec2mel(torch.abs(spec))

        # transform to log scale
        mel = torch.log10(mel + 1e-7)
        # clamp outliers (~10 samples)
        mel = torch.clamp(mel, -4)
        # reshape mel after timestretch
        mel = self.adjust_size(mel)

        # mask mel-spectrogram
        if self.mel_transform:
            mel = self.mel_augment(mel)
        return mel, label
    
    def adjust_size(self, mel: torch.Tensor):
        """Adjust size of the mel after time-stretch, to end up with 
        similar sized spectrograms.
        
        Args:
            mel (torch.Tensor): 2D-representation whose size should be adjusted

        Returns:
            mel (torch.Tensor): mel with adjusted width
        """

        width = mel.size(-1)
        if width >= self.time_bins:
            return mel[..., :self.time_bins]
        else:
            # we want to pad with zeros either on the left or on the roght side of the mel 
            # to avoid a model bias to time stretched audios wihic are always apdded on one side
            insert = random.randint(0, self.time_bins - width)
            if self.split == 'train':
                padded_mel = torch.zeros(1, self.n_mels, self.time_bins)
            else:
                padded_mel = torch.zeros(self.num_chunks, 1, self.n_mels, self.time_bins)
            # insert mel into raw mel mask
            padded_mel[..., insert:insert+width] = mel
            return padded_mel

    def mel_augment(self, mel: torch.Tensor) -> torch.Tensor:
        """Mask batched mel-spectrograms in time and frequency dimension.
        
        Args:
            mel (torch.Tensor): mel-spectrogram
        
        Returns:
            mel (torch.Tensor): masked mel-spectrogram
        """
        # time mask
        num_rows_to_mask = random.randint(1, self.mask_param_time // 2)
        start_row = random.randint(0, mel.shape[-2] - num_rows_to_mask - 1)
        mel[..., start_row:start_row+num_rows_to_mask, :] = 0

        # freq mask
        num_cols_to_mask = random.randint(1, self.mask_param_time)
        start_col = random.randint(0, mel.shape[-1] - num_cols_to_mask - 1)
        mel[..., start_col:start_col+num_cols_to_mask] = 0
        return mel

    def __len__(self):
        return len(self.songlist)
    

def get_loader(data_path: str,
               split: str,
               validation_fold: int = None,
               batch_size: int = 16,
               wav_transform: bool = True, 
               mel_transform: bool = True,
               num_workers: int = 0,
               drop_last: bool = True,
               ) -> DataLoader:
    """Initializes Dataset and creates Dataloader."""

    if split == 'train':
        shuffle = True
    else:
        shuffle = False
        batch_size = batch_size // AUDIO_PARAMS['gtzan']['num_chunks']

    dataset = AudioDataset(data_path=data_path,
                           split=split, 
                           wav_transform=wav_transform, 
                           mel_transform=mel_transform,
                           validation_fold=validation_fold)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader


def get_data_loaders(data_path: str = '../../../data/', 
                     batch_size: int = 16,
                     validation_fold: int = None,
                     num_workers: int = 0,
                     fma: bool = False,
                     ):
    
    trainloader = get_loader(data_path, split='train', batch_size=batch_size, 
                            wav_transform=True, mel_transform=True,
                            validation_fold=validation_fold, 
                            num_workers=num_workers, fma=fma)
    
    validloader = get_loader(data_path, split='valid', batch_size=batch_size, 
                            wav_transform=False, mel_transform=False, 
                            validation_fold=validation_fold, 
                            num_workers=num_workers, fma=fma)
    
    # for corss validation we only need a trian and validation split
    # maybe change to if toy
    """if suffix is not None or fma:
        testloader = get_loader(data_path, split='test', batch_size=batch_size,
                                num_chunks=num_chunks, wav_transform=False, mel_transform=False, 
                                num_workers=num_workers, fma=fma)
        return trainloader, validloader, testloader"""
    
    return trainloader, validloader
