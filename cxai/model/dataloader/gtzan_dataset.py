import os
import sys
import random
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as tat
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from torchaudio_augmentations import (
    RandomApply,
    Compose,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)

from cxai.utils.all import round_down
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
                #cross_valid: bool = False,
                valid_fold: int = None,
                suffix: str = None,
                wav_normalization: str = 'rms',
                mel_normalization: bool = True,
                fma: bool = False,
                ):
        
        self.device = torch.device('mps')
        
        # Data params
        self.data_path = data_path
        self.split = split
        self.fma = fma
        self.genres = CLASS_IDX_MAPPER if not self.fma else CLASS_IDX_MAPPER_FMA
        self.suffix = suffix
        #self.cross_valid = cross_valid
        self.valid_fold = valid_fold

        # Audio params
        ################### tmp workaround for resampling gtzan to 16kHz
        #self.sr_before_resampling = 22050
        #self.num_samples_before_resampling = self.sr_before_resampling * slice_length

        self.sample_rate = sample_rate
        self.slice_length = slice_length
        self.window_size = self.sample_rate * self.slice_length
        self.num_chunks = num_chunks
        self.slice_hop = int(round_down(((29 - self.slice_length) / (self.num_chunks-1)), 1) * self.sample_rate)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.width = 256 if self.sample_rate == 22050 else 128
        # how many bins in mel-spectrogram get masked at max
        self.mask_param_time = 40 if self.sample_rate == 22050 else 20

        # Class specific params
        self.wav_transform = wav_transform
        self.mel_transform = mel_transform
        self.mel_normalization = mel_normalization
        self.wav_normalization = wav_normalization

        self.songlist, self.labels = self.get_songs()

        #self.genre = str.split(self.songlist[0])

        self.mel_converter = self.get_mel_converter()
        #self.to_dB = torchaudio.transforms.AmplitudeToDB()
        #self.resampler = torchaudio.transforms.Resample(22050, self.sample_rate)
        if wav_transform:
            self.wav_augmentation = self.wav_augmentations()
        #if mel_transform:
        #    self.mel_mask = self.mel_augmentations()

        # get dataset statistics of train data to simulate real world scenario whee test data is unknown
        #if not self.mel_normalization:
        #    self.mean, self.std = self.get_data_statistics()

        self.wav2spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None)
        self.spec2mel = torchaudio.transforms.MelScale(sample_rate=self.sample_rate, n_mels=self.n_mels, n_stft=self.n_fft // 2 + 1)
        self.time_stretch = torchaudio.transforms.TimeStretch(hop_length=self.hop_length, n_freq=self.n_fft // 2 + 1)
        

    def get_songs(self, get_stats=False):
        """
        Function to accumulate all songtitles of tha data split for path construction in __get_item__. It creates the songlist
        from different folds for Cross-Validation or from specific, precomputed dataset splits.
        
        Parameters:
        ----------
        self

        Returns:
        -------
        songlist: list
            List of strings. Strings are in the form: 'genre/genre.00000.wav' (e.g, 'jazz/jazz.00023.wav').
        """
        # cross validation
        #if self.cross_valid:
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
        else:
        # no cross validation
            # filtered or fma
            if self.fma:
                list_filename = os.path.join(self.data_path, self.split + '_' + 'split.txt')
                with open(list_filename) as f:
                    lines = f.readlines()
                songlist = []
                labels = []
                for line in lines:
                    song, label = line.strip().split(',')
                    songlist.append(song)
                    labels.append(int(label))
            else:
                # filtered gtzan dataset
                list_filename = os.path.join(self.data_path, self.split + '_' + self.suffix + '.txt' if get_stats==False else 'train' + '_' + self.suffix + '.txt')
                with open(list_filename) as f:
                    lines = f.readlines()
                songlist = []
                for line in lines:
                    name = line.strip()
                    if name == 'jazz/jazz.00054.wav' or name == 'blues/blues.00032.wav': continue
                    songlist.append(name)
                labels = [self.genres[name.split('/')[0]] for name in songlist]
        return songlist, labels
    

    def get_data_statistics(self):
        """
        Function to compute the data statistics of the training data for standardization. Function is only called once
        during initialization of the dataset. This should mimic a real-world scenario, where test data is usually unknown.
        Logic is similar to __get_item__ but performed over all training examples.
        
        Parameters:
        ----------
        self

        Returns:
        -------
        mean: float
            The mean value of all mel-spectrograms to standardize the input for the neural network.
        std: float
            The standarddeviation of the mel-spectrograms.
        """

        # get songlist of train data to compute dataset statistics of train data
        train_songs, _ = self.get_songs(get_stats=True)

        all_mels = []
        for name in train_songs:
            # to an extend similar to __get_item__
            # get audio
            audio_filename = os.path.join(self.data_path, 'genres_original', name)
            wav, _ = torchaudio.load(audio_filename)
            # get slices
            wav = self.get_slice(wav, get_stats=True)
            # perform volume-normalization by normalizing with rms of audio
            wav = AudioDataset.rms_normalizer(wav) if self.wav_normalization=='rms' else AudioDataset.peak_normalizer(wav)
            #wav = self.peak_normalizer(wav)
            # generate mel
            mel = self.mel_converter(wav)
            # transform to DB
            mel = self.to_dB(mel)
            # append to all
            all_mels.append(mel)

        # Stack all spectrograms to calculate mean and std
        all_mels = torch.stack(all_mels).view(-1, mel.size(-2), mel.size(-1))
        # get mean and std across bacth nad time dimension (keep mel bins)
        mean = all_mels.mean()
        std = all_mels.std()

        return mean, std
    
    @staticmethod
    def rms_normalizer(wav, rms_db=0):
        """
        Function to normalize audios in wave-format by rms. (Not usual root-mean-square). Rms is specific for audio data.
        This function essentially scales all audios to some reference dB value such that the loadness of all songs is identical.
        
        Parameters:
        ----------
        self
        wav: torch.Tensor
            Audio snippet in wave-format.
        rms_db: int
            Reference value in dB to scale audio data.

        Returns:
        -------
        scaled_wav: torch.Tensor
            Scaled wav tensor.
        """
        # rms value in dB to rescale the audios
        # convert from db to linear scale
        rms = 10**(rms_db / 20)
        # scaling factor per sample in batch (We want each slice to be on the same db value not the mean of all slices of a song)
        sc = torch.sqrt((wav.size(-1)*rms**2) / (torch.sum(torch.pow(wav, 2), dim=-1, keepdim=True)))
        scaled_wav = wav * sc
        return scaled_wav
    

    @staticmethod
    def peak_normalizer( wav):
        return wav / torch.abs(wav).max(dim=-1, keepdim=True)[0]


    def get_mel_converter(self):
        """
        Function to instanciate the converter from wave-format to mel-spectrograms
        
        Parameters:
        ----------
        self

        Returns:
        -------
        converter: torchaudio.transforms
            Converter which convertes wave files to mel-spectrograms.
        """
        converter = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, window_fn=torch.hann_window, \
                                                         hop_length=self.hop_length, f_min=0.0, f_max=self.sample_rate/2, n_mels=self.n_mels)
        return converter


    def wav_augmentations(self):
        # TODO: If time is left, control these classes and optimize efficiency (write own classes maybe)
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
    

    def mel_augmentations(self):
        mask = Mask(mask_param=self.mask_param)
        return mask


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
        wav = AudioDataset.peak_normalizer(wav)

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
    

    def normalize(mel, epsilon=1e-7):
        '''
        Normalizes e dimensional datapoints to the range [-1,1].
        Data has to be in shape (num_channels, height, width) or (batch_size, num_channels, height, width)
        '''

        mel_min = mel.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        mel_max = mel.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        # normalize each snippet (case 'valid' or 'train' we have several snippets (num_chunks))
        mel = 2*((mel - mel_min) / (mel_max - mel_min + epsilon)) - 1
        
        return mel
    

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
        num_rows_to_mask = random.randint(1, 20 // 2)
        start_row = random.randint(0, mel.shape[-2] - num_rows_to_mask - 1)
        mel[..., start_row:start_row+num_rows_to_mask, :] = 0

        # freq mask
        num_cols_to_mask = random.randint(1, self.mask_param_time)
        start_col = random.randint(0, mel.shape[-1] - num_cols_to_mask - 1)
        mel[..., start_col:start_col+num_cols_to_mask] = 0

        return mel
    

    def __len__(self):
        return len(self.songlist)



class Mask(torch.nn.Module):
    def __init__(
        self,
        mask_param: int = 80,
    ):
        super().__init__()
        self.time_mask = tat.TimeMasking(time_mask_param=mask_param)
        self.frquency_mask = tat.FrequencyMasking(freq_mask_param=mask_param)
        #self.time_stretch = tat.TimeStretch(n_freq = 128)


    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # mask frequency or time or nothing
        rand = random.randint(0, 2)
        if rand == 0:
            masked = self.time_mask(spec)
        elif rand == 1:
            masked = self.frquency_mask(spec)
        else:
            masked = spec
        return masked



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


