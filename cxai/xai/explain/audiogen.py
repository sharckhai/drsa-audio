import os
import sys
import numpy as np
import torch
import librosa

from IPython.display import Audio
import torchaudio
import torchaudio.transforms as T
import torchvision
import torch.nn.functional as F

from cxai.utils.all import get_slice, peak_normalizer, adjust_vol, Loader
from cxai.utils.visualization import plot_spectrogram


class Mel2Audio:
    '''
    Creates audios from mel spectrograms. Loudness of the generated audios is normalized to a fixed dB value.
    '''

    def __init__(self, sample_rate=22050, n_fft=1024, mel_bins=128, blur_kernel=5, sigma=1.0, hop_length=None, slice_length=None, device=torch.device('cpu')) -> None:

        self.device = device
        
        self.n_fft = n_fft
        self.hop_length = self.n_fft//2 if hop_length is None else hop_length
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.slice_length = slice_length

        self.smoother = torchvision.transforms.GaussianBlur(blur_kernel, sigma=sigma)



    def transform(self, heatmap, orig_mel, phase, percentile=None):
        """
        Masks the Mel-spectrogram with the given heatmap and transforms the resulting mel-spec back to wave-form audio.

        1. Axtract mel and Phases of audio
        2. Generate the heatmap mask (mel)
        3. Apply mask
        4. Transform into spectrogram
        5. Apply original phases to the signal
        6. Inverse STFT
        """

        # build mask
        heatmap_mask = Mel2Audio.generate_mask(heatmap, self.smoother, percentile)

        # mask orinal mel
        mel = orig_mel * heatmap_mask

        # tranform masked mel to spectrogram
        inv_mel = librosa.feature.inverse.mel_to_stft(mel, sr=self.sample_rate, n_fft=self.n_fft, norm=None, htk=True)

        # add phase and perform inverse STFT
        recovered_wav = librosa.istft(inv_mel*phase, hop_length=self.n_fft//2, n_fft=self.n_fft)

        return recovered_wav
    

    def transform_audio(self, wav):
        '''
        Loads the original audio and constructs Mel-spectrogram to apply the heatmap mask on.
        During stft and mel transformation, we save the phases of the signal for the inverse transformation.
        '''

        wav = wav.cpu().numpy() if isinstance(wav, torch.Tensor) else wav

        # build complex spectrogram
        spec_cplx = librosa.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, pad_mode='reflect')
        # extract the magnitude spectrogram and phases
        mag, phase = librosa.magphase(spec_cplx)
        # transfrom the magnitude spectrogram into a mel-spectrogram (of similar form as the orioginal input_sample to extract the heatmaps)
        # we could also just extract the phases and use the original mel-spectrogram for the inverse transformation
        
        # TODO: change this, dont square just pass the power argument to the function --> way faster
        
        mel = librosa.feature.melspectrogram(S=mag**2, sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.mel_bins, htk=True, norm=None)       
        ############ ggf hier **2 weg, und bei inverse.mel_to_stft power=1 oder power=2 dazu, geht viel schneller!!!

        return mel[..., 1:-2].squeeze(), phase[..., 1:-2].squeeze()
    

    def transform_audio_from_file(self, path_to_sample, startpoint=None):

        wav, _ = torchaudio.load(path_to_sample)
        wav = wav.requires_grad_(False)

        if startpoint is not None:
            wav = get_slice(wav, slice_length=self.slice_length, start_point=startpoint).squeeze().numpy()

        wav = wav.cpu().numpy() if isinstance(wav, torch.Tensor) else wav

        return self.transform_audio(wav)


    @staticmethod
    def generate_mask(heatmap, smoother=None, percentile=None):

        heatmap = heatmap if isinstance(heatmap, torch.Tensor) else torch.tensor(heatmap).requires_grad_(False)

        # set negatives to 0
        heatmap_pos = F.relu(heatmap)

        # get rid of noise
        if percentile:
            percentile_value = np.percentile(heatmap_pos, percentile)
            heatmap_pos = heatmap_pos*(heatmap_pos > percentile_value)

        print(heatmap_pos.shape)

        # smooth heatmap mask
        heatmap_blurred = smoother(heatmap_pos)
        #heatmap_blurred = heatmap_filtered

        # set max value to 1 (normalization)
        #heatmap_blurred = (heatmap_blurred - torch.min(heatmap_blurred)) / (torch.max(heatmap_blurred) - torch.min(heatmap_blurred))

        return heatmap_blurred.squeeze().numpy()

    

    def transform_mel(self, mel, path_to_sample, startpoint):
        """
        Purpose of this function is to evaluate the quality of the invers transformation.
        (To compare the original wav with the geneerated wav form the mel)
        """

        _, phase = self.transform_audio(path_to_sample, startpoint)

        inv_mel = librosa.feature.inverse.mel_to_stft(mel, sr=self.sample_rate, n_fft=self.n_fft, norm=False, htk=True)

        final_waveform = librosa.istft(inv_mel*phase, hop_length=self.n_fft//2, n_fft=self.n_fft)

        return final_waveform
    

    def make_audios(self, sample_info, orig_wav, startpoint=None, num_concepts=4, percentile=50, path_to_sample=None, sample_idx=0):

        assert orig_wav is not None or path_to_sample is not None, 'please provide either an audio example or path to audio file'

        print(orig_wav.dtype)

        if path_to_sample:
            assert startpoint != None, 'if path to audio, please provide startpoint for audio snippet'
            mel, phase = self.transform_audio_from_file(path_to_sample, startpoint)
        else:
            orig_wav = peak_normalizer(orig_wav)
            mel, phase = self.transform_audio(orig_wav)

        audios = []

        #audios.append(adjust_vol(orig_wav, torch.tensor(self.transform(sample_info['standard_heatmap'], mel, phase, percentile=50))))
        #standard_heatmap = rms_normalizer(torch.tensor(self.transform(sample_info['standard_heatmap'], mel, phase, percentile=50)))
        standard_heatmap = peak_normalizer(torch.tensor(self.transform(sample_info['standard_heatmaps'][sample_idx], mel, phase, percentile=50)))
        #audios.append(adjust(orig_wav, standard_heatmap))
        audios.append(standard_heatmap)

        for subspace_idx in range(num_concepts):
            # generate one subspace audio
            wav = self.transform(sample_info['subspace_heatmaps'][sample_idx][subspace_idx][None], mel, phase, percentile=percentile)
            # adjust volume to original audio
            #audios.append(adjust(orig_wav, peak_normalizer(torch.tensor(wav))))
            audios.append(peak_normalizer(torch.tensor(wav)))

        return audios



class Mel2AudioTs:
    '''
    Creates audios from mel spectrograms. Loudness of the generated audios is normalized to a fixed dB value.
    '''

    def __init__(self, sample_rate=16000, n_fft=800, mel_bins=128, blur_kernel=5, sigma=1.0, hop_length=None, slice_length=None, device=torch.device('cpu')) -> None:

        self.device = device
        
        self.n_fft = n_fft
        self.hop_length = self.n_fft//2 if hop_length is None else hop_length
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.slice_length = slice_length

        self.smoother = torchvision.transforms.GaussianBlur(blur_kernel, sigma=sigma)
        self.loader = Loader()



    def transform(self, heatmap, orig_mel, phase, percentile=None):
        """
        Masks the Mel-spectrogram with the given heatmap and transforms the resulting mel-spec back to wave-form audio.

        1. Axtract mel and Phases of audio
        2. Generate the heatmap mask (mel)
        3. Apply mask
        4. Transform into spectrogram
        5. Apply original phases to the signal
        6. Inverse STFT
        """

        # build mask
        heatmap_mask = Mel2Audio.generate_mask(heatmap, self.smoother, percentile)

        # mask orinal mel
        mel = orig_mel * heatmap_mask

        # tranform masked mel to spectrogram
        inv_mel = librosa.feature.inverse.mel_to_stft(mel, sr=self.sample_rate, n_fft=self.n_fft, norm=None, htk=True, power=1)
        #inv_mel = librosa.feature.inverse.mel_to_stft(mel, sr=16000, n_fft=self.n_fft, norm=None, htk=True, power=1) ##################### temporary fix

        # add phase and perform inverse STFT
        recovered_wav = librosa.istft(inv_mel*phase, hop_length=self.hop_length, n_fft=self.n_fft)

        return recovered_wav
    

    def transform_audio(self, wav) -> np.ndarray:
        '''
        Loads the original audio and constructs Mel-spectrogram to apply the heatmap mask on.
        During stft and mel transformation, we save the phases of the signal for the inverse transformation.
        '''

        wav = wav if isinstance(wav, torch.Tensor) else torch.tensor(wav)

        wav, mag, phase, mel = self.loader.transform_wav(wav, return_all=True)

        #spec_cplx = librosa.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, pad_mode='reflect')
        # extract the magnitude spectrogram and phases
        #mag, phase = librosa.magphase(spec_cplx)

        # transfrom the magnitude spectrogram into a mel-spectrogram (of similar form as the orioginal input_sample to extract the heatmaps)
        # we could also just extract the phases and use the original mel-spectrogram for the inverse transformation
        #mel = librosa.feature.melspectrogram(S=mag, sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.mel_bins, htk=True, norm=None, center=True, pad_mode='reflect', power=1)
        #mel = np.log10(mel+1e-7)

        return mel[..., :256].squeeze(), phase[..., :256].squeeze()
    

    def transform_audio_from_file(self, path_to_sample, startpoint=None):

        wav, _ = torchaudio.load(path_to_sample)
        wav = wav.requires_grad_(False)

        if startpoint is not None:
            wav = get_slice(wav, slice_length=self.slice_length, start_point=startpoint).squeeze().numpy()

        wav = wav.cpu().numpy() if isinstance(wav, torch.Tensor) else wav

        return self.transform_audio(wav)


    @staticmethod
    def generate_mask(heatmap, smoother=None, percentile=None):

        heatmap = heatmap if isinstance(heatmap, torch.Tensor) else torch.tensor(heatmap).requires_grad_(False)

        # set negatives to 0
        heatmap_pos = F.relu(heatmap)

        # get rid of noise
        if percentile:
            percentile_value = np.percentile(heatmap_pos, percentile)
            heatmap_pos = heatmap_pos*(heatmap_pos > percentile_value)

        # smooth heatmap mask
        heatmap_blurred = smoother(heatmap_pos)
        #heatmap_blurred = heatmap_filtered

        # set max value to 1 (normalization)
        #heatmap_blurred = (heatmap_blurred - torch.min(heatmap_blurred)) / (torch.max(heatmap_blurred) - torch.min(heatmap_blurred))

        return heatmap_blurred.squeeze().numpy()

    

    def transform_mel(self, mel, path_to_sample, startpoint):
        """
        Purpose of this function is to evaluate the quality of the invers transformation.
        (To compare the original wav with the geneerated wav form the mel)
        """

        _, phase = self.transform_audio(path_to_sample, startpoint)

        inv_mel = librosa.feature.inverse.mel_to_stft(mel, sr=self.sample_rate, n_fft=self.n_fft, norm=False, htk=True)

        final_waveform = librosa.istft(inv_mel*phase, hop_length=self.n_fft//2, n_fft=self.n_fft)

        return final_waveform
    

    def make_audios(self, sample_info, orig_wav, startpoint=None, num_concepts=4, percentile=50, path_to_sample=None):

        assert orig_wav is not None or path_to_sample is not None, 'please provide either an audio example or path to audio file'

        if path_to_sample:
            assert startpoint != None, 'if path to audio, please provide startpoint for audio snippet'
            mel, phase = self.transform_audio_from_file(path_to_sample, startpoint)
        else:
            orig_wav = peak_normalizer(orig_wav)
            mel, phase = self.transform_audio(orig_wav)

        audios = []

        standard_heatmap = peak_normalizer(torch.tensor(self.transform(sample_info['standard_heatmap'], mel, phase, percentile=50)))
        audios.append(adjust_vol(orig_wav, standard_heatmap))

        for subspace_idx in range(num_concepts):
            # generate one subspace audio
            wav = self.transform(sample_info['subspace_heatmaps'][subspace_idx:subspace_idx+1], mel, phase, percentile=percentile)
            # adjust volume to original audio
            audios.append(adjust_vol(orig_wav, peak_normalizer(torch.tensor(wav))))

        return audios
    
