import random
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as tat
from torch.utils.data import Dataset, DataLoader
import numpy as np
import timeit
import os
from torchaudio_augmentations import (
    RandomApply,
    Compose,
    RandomResizedCrop,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)


class ToyDataset(Dataset):
    '''
    New dataset class for toy data since subclass of custom dataset doesnt allow rewriting __get_item__ method
    
    '''
    def __init__(self, 
                data_path: str,
                split: str,
                sample_rate: int = 16000,
                n_mels: int = 64, 
                n_fft: int = 480, 
                mask_param: int = 10,
                wav_transform: bool = True, 
                mel_transform: bool = True,
                ):
        random.seed(42)

        # Data params
        self.data_path = data_path
        self.split = split
        self.dataclasses = {'class1': 0, 'class2': 1}    
        self.mel_transform = mel_transform
        self.wav_transform = wav_transform

        # audio params
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.mask_param = mask_param
        self.width = 64

        # Class specific params
        self.song_list, self.labels = self.get_songs()
        self.mel_converter = self.get_mel_converter()

        if wav_transform:
            self.wav_augmentation = self.wav_augmentations()

        self.wav2spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.n_fft//2, power=None)
        self.spec2mel = torchaudio.transforms.MelScale(sample_rate=self.sample_rate, n_mels=self.n_mels, n_stft=self.n_fft // 2 + 1)
        self.time_stretch = torchaudio.transforms.TimeStretch(hop_length=self.n_fft//2, n_freq=self.n_fft // 2 + 1)


    def get_songs(self):
        # get song names (e.g., 'jazz/jazz.00023.wav') and wirte them to list to load them from files later
        list_filename = os.path.join(self.data_path, self.split + '_split.txt')
        with open(list_filename) as f:
            lines = f.readlines()

        songlist = []
        labels = []
        for line in lines:
            line = line.strip()
            songlist.append(line)
            labels.append(self.dataclasses[line.split('/')[0]])

        #songlist = [line.strip() for line in lines]
        return songlist, labels
    

    def rms_normalizer(self, wav, rms_db=0):
        # rms value in dB to rescale the audios
        # convert from db to linear scale
        rms = 10**(rms_db / 20)
        # scaling factor per sample in batch (We want each slice to be on the same db value not the mean of all slices of a song)
        sc = torch.sqrt((wav.size(-1)*rms**2) / (torch.sum(torch.pow(wav, 2), dim=-1, keepdim=True)))
        return wav * sc
    
    @staticmethod
    def peak_normalizer( wav):
        return wav / torch.max(torch.abs(wav))


    def get_mel_converter(self):
        converter = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, f_min=0.0, f_max=self.sample_rate/2, n_mels=self.n_mels)
        return converter
    
    def wav_augmentations(self):
        # TODO: If time is left, control these classes and optimize efficiency (write own classes maybe)
        transforms = [
            RandomApply([Gain(min_gain=-12, max_gain=3)], p=0.5),
            #RandomApply([PitchShift(n_samples=self.sample_rate, sample_rate=self.sample_rate, pitch_shift_max=12, pitch_shift_min=-12)], p=0.3),
            #RandomApply([HighLowPass(sample_rate=self.sample_rate, 
            #                        lowpass_freq_low = 1400,
            #                        lowpass_freq_high = 4000,
            #                        highpass_freq_low = 200,
            #                        highpass_freq_high = 1400,)], p=0.4),
            RandomApply([Delay(sample_rate=self.sample_rate, volume_factor=0.5, min_delay=50, max_delay=300)], p=0.4),
            RandomApply([Reverb(sample_rate=self.sample_rate)], p=0.3),
            RandomApply([Noise(min_snr=1e-3, max_snr=1e-1)], p=0.3),
        ]
        return Compose(transforms=transforms)


    def __getitem__(self, index):
        # get name of sample for path
        name = self.song_list[index]
        label = self.labels[index]

        # get audio
        wav, _ = torchaudio.load(os.path.join(self.data_path, name))

        # perform volume-normalization by normalizing by peak
        wav = ToyDataset.peak_normalizer(wav)

        # data augmentation
        if self.wav_transform:
            wav = self.wav_augmentation(wav)

        # generate mel (directly transform to mel sinze we dont need cplx spectrogram here because we perform nop time stretching)
        #mel = self.mel_converter(wav)

        # PIPELINE  
        spec = self.wav2spec(wav) # complex spectrogram

        #if self.mel_transform: 
        #        spec = self.time_stretch(spec, round(random.uniform(0.8,1.2), 3))

        mel = self.spec2mel(torch.abs(spec))

        # transform to log scale
        mel = torch.log10(mel + 1e-7)

        # clamp outliers (~10 samples)
        #mel = torch.clamp(mel, -4)

        # normalize each snippet to [-1,1]
        #mel = AudioDataset.normalize(mel)

        # reshape mel after timestretch
        mel = self.adjust_size(mel)

        #mel = (mel - mel.max()) / (mel.max() - mel.min())

        # reshape mel after timestretch
        #mel = mel[..., 1:-2]
    
        # mask mel-spectrogram
        if self.mel_transform:
            mel = self.mel_augment(mel)


        return mel, label
    
    
    def mel_augment(self, mel):

        i = random.randint(1,2)
        if i==1:
            # freq mask
            num_rows_to_mask = random.randint(1, self.mask_param // 2 + 1)
            start_row = random.randint(0, mel.shape[-2] - num_rows_to_mask + 1)
            mel[..., start_row:start_row+num_rows_to_mask, :] = 0
        else:
            # time mask
            num_cols_to_mask = random.randint(1, self.mask_param + 1)
            start_col = random.randint(0, mel.shape[-1] - num_cols_to_mask + 1)
            mel[..., start_col:start_col+num_cols_to_mask] = 0

        return mel
    

    def adjust_size(self, mel):

        w = mel.size(-1)

        if w >= self.width:
            return mel[..., :self.width]
        else:
            insert = random.randint(0, self.width-w)

            full_mel = torch.zeros(1,self.n_mels,self.width)

            full_mel[..., insert:insert+w] = mel
            return full_mel
    
    
    def __len__(self):
        return len(self.song_list)



##### loader for toy data sets
def get_toy_loader(data_path: str,
                split: str,
                batch_size: int = 16,
                sample_rate: int = 16000,
                n_mels: int = 64, 
                n_fft: int = 480, 
                mask_param: int = 20,
                wav_transform: bool = True, 
                mel_transform: bool = True,
                num_workers: int = 0,
                drop_last: bool = False
                ):
    shuffle = True if (split == 'train') else False
    #batch_size = batch_size if (split == 'train') else (batch_size // num_chunks)

    dataset = ToyDataset(data_path=data_path,
                            split=split, 
                            sample_rate=sample_rate,
                            n_mels=n_mels, 
                            n_fft=n_fft, 
                            mask_param=mask_param,
                            wav_transform=wav_transform, 
                            mel_transform=mel_transform)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return loader



def get_toydata_loaders(data_path='../../Data/', sample_rate=16000, batch_size=16, n_mels=64, n_fft=480, mask_param=20, wav_transform=False, mel_transform=False):
    
    trainloader = get_toy_loader(data_path, split='train', batch_size=batch_size, sample_rate=sample_rate,
                            n_mels=n_mels, n_fft=n_fft, mask_param=mask_param, wav_transform=wav_transform, mel_transform=mel_transform) ########## no data augmentation
    validloader = get_toy_loader(data_path, split='valid',  batch_size=batch_size,  sample_rate=sample_rate, 
                            n_mels=n_mels, n_fft=n_fft, mask_param=mask_param, wav_transform=False, mel_transform=False)
    testloader = get_toy_loader(data_path, split='test', n_mels=n_mels, batch_size=batch_size, sample_rate=sample_rate,
                            n_fft=n_fft, mask_param=mask_param, wav_transform=False, mel_transform=False)
    return trainloader, validloader, testloader
