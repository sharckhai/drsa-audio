import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio_augmentations import (
    RandomApply,
    Compose,
    Noise,
    Gain,
    HighLowPass,
    PitchShift,
)

from cxai.utils.sound import round_down, peak_normalizer, normalize
from cxai.utils.constants import CLASS_IDX_MAPPER


class AudioDataset(Dataset):
    '''
    TensorDataset with support of transforms

    - Loads sliced audios from .wav files
    - applies audio transforms such as noise, pitch shifts etc
    - transforms to spectrogram
    - apllies spectrogram augmentations (time stretch, time mask, frq mask)
    - transforms to mel and normalizes by frequency bin (mel bin)

    '''
    def __init__(self, 
                data_path: str,
                split: str,
                sample_rate: int = 16000,
                n_mels: int = 128, 
                n_fft: int = 800, 
                hop_length: int = 360,
                mask_param: int = 40,
                slice_length: int = 6,
                num_chunks: int = 4,
                wav_transform: bool = True, 
                mel_transform: bool = True,
                valid_fold: int = 1,
                suffix: str = None,
                wav_normalization: str = 'rms',
                mel_normalization: bool = True,
                ):
        
        self.device = torch.device('mps')
        
        # Data params
        self.data_path = data_path
        self.split = split
        self.genres = CLASS_IDX_MAPPER
        self.suffix = suffix
        self.valid_fold = valid_fold

        # audio params
        self.sample_rate = sample_rate
        self.slice_length = slice_length
        self.window_size = self.sample_rate * self.slice_length
        self.num_chunks = num_chunks
        self.slice_hop = int(round_down(((29 - self.slice_length) / (self.num_chunks-1)), 1) * self.sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.width = 256 if self.sample_rate == 22050 else 128
        
        # Class specific params
        self.mask_param_time = mask_param
        self.wav_transform = wav_transform
        self.mel_transform = mel_transform
        self.mel_normalization = mel_normalization
        self.wav_normalization = wav_normalization

        # songlist and transforms
        self.songlist, self.labels = self.get_songs()
        if wav_transform:
            self.wav_augmentation = self.wav_augmentations()
        self.wav2spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None)
        self.spec2mel = torchaudio.transforms.MelScale(sample_rate=self.sample_rate, n_mels=self.n_mels, n_stft=self.n_fft // 2 + 1)
        self.time_stretch = torchaudio.transforms.TimeStretch(hop_length=self.hop_length, n_freq=self.n_fft // 2 + 1)
        

    def get_songs(self, get_stats=False) -> list:
        """
        Function to accumulate all songtitles of tha data split for path construction in __get_item__. It creates the songlist
        from different folds for Cross-Validation or from specific, precomputed dataset splits.
        -----
        Returns (list): List of strings. Strings are in the form: 'genre/genre.00000.wav' (e.g, 'jazz/jazz.00023.wav').
        """
        if self.valid_fold is not None:
            if self.split == 'train' or get_stats:
                # case train and cross-validation: concatenate all folds together exept of the validation fold
                songlist = []
                labels = []
                for fold in range(1,6):
                    if fold != self.valid_fold:
                        list_filename = os.path.join(self.data_path, f'fold_{fold}.txt')
                        with open(list_filename) as f:
                            lines = f.readlines()
                        # extend songlist
                        songlist.extend([line.strip() for line in lines])
                        labels.extend([self.genres[name.split('/')[0]] for name in songlist])
            else:
                list_filename = os.path.join(self.data_path, f'fold_{self.valid_fold}.txt')
                with open(list_filename) as f:
                    lines = f.readlines()
                songlist = [line.strip() for line in lines]
                labels = [self.genres[name.split('/')[0]] for name in songlist]

        return songlist, labels


    def wav_augmentations(self):
        transforms = [
            RandomApply([Gain(min_gain=-12, max_gain=3)], p=0.5),
            RandomApply([PitchShift(n_samples=self.window_size, sample_rate=self.sample_rate, pitch_shift_max=12, pitch_shift_min=-12)], p=0.3),
            RandomApply([HighLowPass(sample_rate=self.sample_rate, 
                                    lowpass_freq_low = 1400,
                                    lowpass_freq_high = 4000,
                                    highpass_freq_low = 200,
                                    highpass_freq_high = 1400,)], p=0.4),
            RandomApply([Noise(min_snr=1e-3, max_snr=1e-1)], p=0.3),
        ]
        return Compose(transforms=transforms)
    

    def get_slice(self, wav, get_stats=False):
        if self.split != 'train' or get_stats:
            # unfold along the time dimension
            wav = wav[:, :29*self.sample_rate].unfold(1, self.window_size, self.slice_hop).reshape(-1, 1, self.window_size)
        else:
            # if train we sample a random part of the wav with length = slice_length
            random_index = random.randint(0, wav.size(1) - self.window_size - 1)
            wav = wav[:, random_index : random_index + self.window_size]
        return wav


    def __getitem__(self, index):

        # get name of sample for path
        name = self.songlist[index]
        #label = self.labels[index]
        label = self.genres[name.split('/')[0]]

        # get audio
        wav, _ = torchaudio.load(os.path.join(self.data_path, 'genres_original', name)) #if not self.fma else os.path.join(self.data_path, 'fma_small', name))
        wav = wav.requires_grad_(False)#.to(self.device)

        # get slice (adjust audio length)
        wav = self.get_slice(wav)

        # perform volume-normalization by normalizing with rms of audio
        wav = peak_normalizer(wav)

        # data augmentation
        if self.wav_transform:
            wav = self.wav_augmentation(wav)

        # PIPELINE  
        spec = self.wav2spec(wav) # complex spectrogram

        if self.mel_transform: 
                spec = self.time_stretch(spec, round(random.uniform(0.8,1.2), 3))

        mel = self.spec2mel(torch.abs(spec))

        # transform to log scale
        mel = torch.log10(mel + 1e-7)

        # clamp outliers (~10 samples)
        mel = torch.clamp(mel, -4)

        # normalize each snippet to [-1,1]
        #mel = AudioDataset.normalize(mel)

        # reshape mel after timestretch
        mel = self.adjust_size(mel)

        # mask mel-spectrogram
        if self.mel_transform:
            mel = self.mel_augment(mel)

        return mel, label
    

    def adjust_size(self, mel):

        w = mel.size(-1)

        if w >= self.width:
            return mel[..., :self.width]
        else:
            insert = random.randint(0, self.width-w)
            full_mel = torch.zeros(1,self.n_mels,self.width) if self.split == 'train' else torch.zeros(self.num_chunks, 1,self.n_mels,self.width)
            full_mel[..., insert:insert+w] = mel
            return full_mel


    def mel_augment(self, mel):
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
                suffix: str = None,
                batch_size: int = 16,
                sample_rate: int = 22050,
                n_mels: int = 128, 
                n_fft: int = 1024, 
                hop_length: int = 512,
                slice_length: int = 6,
                num_chunks: int = 4,
                wav_transform: bool = True, 
                mel_transform: bool = True,
                valid_fold: int = None,
                num_workers: int = 0,
                drop_last: bool = True,  ################## tmporary false
                wav_normalization: str = None, 
                mel_normalization: bool = True,
                fma: bool = False,
                ):
    shuffle = True if (split == 'train') else False
    batch_size = batch_size if (split == 'train') else (batch_size // num_chunks)

    dataset = AudioDataset(data_path=data_path,
                            split=split, 
                            sample_rate=sample_rate,
                            n_mels=n_mels, 
                            n_fft=n_fft, 
                            hop_length=hop_length,
                            slice_length=slice_length,
                            num_chunks=num_chunks,
                            wav_transform=wav_transform, 
                            mel_transform=mel_transform,
                            valid_fold=valid_fold,
                            wav_normalization=wav_normalization, 
                            mel_normalization=mel_normalization,
                            suffix=suffix,
                            fma=fma)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return loader



def get_data_loaders(data_path: str = '../../../data/', 
                     batch_size: int = 16,
                     sample_rate: int = 22050,
                     n_mels: int = 128, 
                     n_fft: int = 1024, 
                     hop_length: int = 512, 
                     slice_length: int = 6, 
                     num_chunks: int = 4, 
                     valid_fold: int = None,
                     wav_normalization: str = None, 
                     mel_normalization: bool = True,
                     num_workers: int = 0,
                     suffix: str = None,
                     fma: bool = False,
                     ):
    
    trainloader = get_loader(data_path, split='train', suffix=suffix, batch_size=batch_size, n_mels=n_mels,
                            n_fft=n_fft, hop_length=hop_length, slice_length=slice_length, sample_rate=sample_rate,
                            num_chunks=num_chunks, wav_transform=True, mel_transform=True,
                            valid_fold=valid_fold, wav_normalization=wav_normalization, 
                            mel_normalization=mel_normalization, num_workers=num_workers, fma=fma)
    
    validloader = get_loader(data_path, split='valid', suffix=suffix, batch_size=batch_size, n_mels=n_mels, 
                            n_fft=n_fft, hop_length=hop_length, slice_length=slice_length, sample_rate=sample_rate,
                            num_chunks=num_chunks, wav_transform=False, mel_transform=False, 
                            valid_fold=valid_fold, wav_normalization=wav_normalization, 
                            mel_normalization=mel_normalization, num_workers=num_workers, fma=fma)
    
    # for corss validation we only need a trian and validation split
    if suffix is not None or fma:
        testloader = get_loader(data_path, split='test', suffix=suffix, batch_size=batch_size, n_mels=n_mels,
                                n_fft=n_fft, hop_length=hop_length, slice_length=slice_length, sample_rate=sample_rate,
                                num_chunks=num_chunks, wav_transform=False, mel_transform=False, 
                                wav_normalization=wav_normalization, mel_normalization=mel_normalization, num_workers=num_workers, fma=fma)
        
        return trainloader, validloader, testloader
    
    return trainloader, validloader


