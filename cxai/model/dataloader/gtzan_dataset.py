import random
from pathlib import Path
from typing import Tuple

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
        mel_augment: bool = True,
        device: str | torch.device = torch.device('mps')
    ) -> None:
        """Init dataset class for audio loading and augmentation.
        
        Args:
            data_path (str | Path): Path to data folder.
            split (str): Options ['train', 'validation'].
            validation_fold (int): Defines which fold is used for validation.
            mask_param (int): Maximum number of bins masked per dimension of the mel.
            wav_augment (bool): Flag to switch on/off waveform transformations.
            mel_augmel_augmentent (bool): Flag to switch on/off mel transformations.
            device (str | torch.device): Device.
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
        self.mel_augment = mel_augment
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
            RandomApply(
                [PitchShift(
                    n_samples=self.window_size, 
                    sample_rate=self.sample_rate, 
                    pitch_shift_max=12, 
                    pitch_shift_min=-12)], 
                    p=0.3
                ),
            RandomApply(
                [HighLowPass(
                    sample_rate=self.sample_rate, 
                    lowpass_freq_low = 1400,
                    lowpass_freq_high = 4000,
                    highpass_freq_low = 200,
                    highpass_freq_high = 1400,)], 
                    p=0.4
                ),
            RandomApply(
                [Noise(min_snr=1e-3, max_snr=1e-1)], 
                p=0.3
            ),
        ]
        return Compose(transforms=transforms)
    
    def get_slice(self, wav: torch.Tensor, label: torch.Tensor):
        """Extract slice/slices from original waveform audio.

        If we are in train phase (split==train), we slice randomly one snippet out of the original audio.
        If we are in validation phase, to avoid inconsistencies in validation accuracy, we always use all data
        for validation. This means we slice the waveform audio in num_chunks disjunct audio snippets.
        
        Args:
            wav (troch.Tensor): Waveform audio.

        Returns:
            sliced_wavs (troch.Tensor): Waveform audios of shorter length (slices).
        """
        if self.split == 'train':
            # if train we sample a random part of the wav with length = slice_length
            random_index = random.randint(0, wav.size(1) - self.window_size - 1)
            sliced_wavs = wav[:, random_index : random_index + self.window_size]
        else:
            # unfold along the time dimension
            sliced_wavs = wav[:, :29*self.sample_rate].unfold(
                1, 
                self.window_size,
                self.slice_hop
            )
            sliced_wavs = sliced_wavs.reshape(-1, 1, self.window_size)
            # we have to duplicate the label as often as we have created slices for validation
            label = label.repeat(self.num_chunks)
        return sliced_wavs, label

    def __getitem__(self, index):
        """Loads waveform audio and transforms it into a log-mel-spectrogram.

        Pipeline:
        1. Load waveform audio
        2. Slice audio into smaller chunks (one ranom chunk during trainphase, several chunks during valid phase).
        3. Normalize audio snipptes by peak.
        4. Transform into complex spectrogram.
        5. Apply time stretching if mel_augment==True.
        6. Project magnitude spectrogram onto mel scale. 
        7. Project amplitudes onto the log10 scale.
        8. Clamp outlier amplitudes.
        9. Adjust size of mel spectrograms after time stretch.
        10. Mask mel spectrogram in time and frequency dimension.

        Args:
            index (int): Index of loader instance.

        Returns:
            mel (torch.Tensor): Log-mel-spcetrogram.
        """
        # songpath and label
        path_to_song = self.songlist[index]
        label = torch.Tensor([self.labels[index]]).requires_grad_(False)

        # load audio waveform
        wav, _ = torchaudio.load(path_to_song)
        wav = wav.requires_grad_(False)#.to(self.device)
        # slice audio
        wav, labels = self.get_slice(wav, label)
        # perform volume-normalization of waveform by peak
        wav = peak_normalizer(wav) # TODO: check if this works with batch for each sample in the batch individually
        # data augmentation
        if self.wav_augment:
            wav = self.wav_augmentation(wav)

        # perform stft  
        spec = self.wav2spec(wav) # complex spectrogram
        # perform time stretching
        if self.mel_augment: 
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
        if self.mel_augment:
            mel = self.mel_augmentation(mel)
        return mel, labels
    
    def adjust_size(self, mel: torch.Tensor) -> torch.Tensor:
        """Adjust size of the mel after time-stretch, to end up with 
        similar sized spectrograms.
        
        Args:
            mel (torch.Tensor): 2D-representation whose size should be adjusted.

        Returns:
            mel_adjusted (torch.Tensor): Mel with adjusted width.
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

    def mel_augmentation(self, mel: torch.Tensor) -> torch.Tensor:
        """Mask batched mel-spectrograms in time and frequency dimension.
        
        Args:
            mel (torch.Tensor): Mel-spectrogram or batch.
        
        Returns:
            mel (torch.Tensor): Masked mel-spectrogram or batch.
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
    

def get_loader(
    data_path: str,
    split: str,
    validation_fold: int = None,
    batch_size: int = 16,
    wav_transform: bool = True, 
    mel_transform: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """Initializes Dataset and creates Dataloader.
    
    Args:
        data_path (str): Patrh to data root.
        split (str): 'train' or 'valid'.
        validation_fold (int): Which fold to use for validation.
        batch_size (int): Batch size.
        wav_transform (bool): Flag to control waveform augmentation during dataloading.
        mel_transform (bool): Flag to control mel augmentation during dataloading.
        num_workers (int): Number of workers for dataloading parallelization.
        drop_last (bool): Drop last batch if uncomplete.
    
    Returns:
        loader (DataLoader): Dataloader.
    """
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
        batch_size = batch_size // AUDIO_PARAMS['gtzan']['num_chunks']

    dataset = AudioDataset(
        data_path=data_path,
        split=split, 
        wav_transform=wav_transform, 
        mel_transform=mel_transform,
        validation_fold=validation_fold
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return loader


def get_data_loaders(
    data_path: str = '../../../data/', 
    batch_size: int = 16,
    validation_fold: int = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Init dataloaders for model training.
    
    Args:
        data_path (str): Patrh to data root.
        batch_size (int): Batch size.
        validation_fold (int): Which fold to use for validation.
        num_workers (int): Number of workers for dataloading parallelization.

    Returns:
        tuple: A tuple containing:
            - trainloader (DataLoader): Dataloader for training.
            - validloader (DataLoader): Dataloader for validation.
    """
    trainloader = get_loader(
        data_path, 
        split='train', 
        batch_size=batch_size, 
        wav_transform=True, 
        mel_transform=True,
        validation_fold=validation_fold, 
        num_workers=num_workers, 
    )
    validloader = get_loader(
        data_path, 
        split='valid', 
        batch_size=batch_size, 
        wav_transform=False, 
        mel_transform=False, 
        validation_fold=validation_fold, 
        num_workers=num_workers, 
    )
    return trainloader, validloader
