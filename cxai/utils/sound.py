import numpy as np
import torch
import torchaudio

from cxai.utils.utilities import round_down


def get_slice(
    wav: torch.Tensor, 
    slice_length: int = 6, 
    start_point: int = 0, 
    num_chunks: int = 1, 
    sample_rate: int = 16000
) -> torch.Tensor:
    """Extracts evenly spaced slices (if num_chunks > 1) from an audio signal (wav).

    Args:
        wav (torch.Tensor): Audio signal in waveform.
        slice_length (int, optional): Slice length in seconds of each extracted snippet from the music sample.
        start_point (int, optional): Start point in seconds to extract a snippet (only used if num_chunks==1).
        num_chunks (int, optional): Number of snippets that should be extracted from the audio.
        sample_rate (int, optional): Sample rate of the audio file.

    Returns:
        wav_sample_sliced (torch.Tensor): Sliced instance. Can be single snippet or multiple snippets.
    """
    wav = torch.tensor(wav) if not isinstance(wav, torch.Tensor) else wav
    # calculate window size of an audio snippet in time bins
    window_size = int(slice_length * sample_rate)
    
    if num_chunks > 1:
        # slice audio waveforms in several chunks
        # we set minimum length of the audios to 29s (shortest audio is ~29.3s)
        hop = int(round_down(((29 - slice_length) / (num_chunks - 1)), 1) * sample_rate)
        # unfold along the time dimension
        wav_sample_sliced = wav[:, :29*sample_rate].unfold(1, window_size, hop).reshape(-1, 1, window_size)
        assert wav_sample_sliced.shape[0] == num_chunks, 'not equal num_chunks'
    else:
        # only extract one slice from the waveform audio
        start_sample = int(start_point * sample_rate)
        assert start_point <= wav.size(1) - window_size, f'Start_point has to be in range [{0},{wav.size(1) - window_size}]'
        wav_sample_sliced = wav[:, start_sample:start_sample+window_size]

    return wav_sample_sliced


def rms_normalizer(wav: torch.Tensor, rms_db: int = 0) -> torch.Tensor:
    """Normalize waveform audios by rms. 
    
    This function essentially scales all audios to some reference dB value such that the loudness of all songs is identical.
    
    Args:
        wav (torch.Tensor): Audio snippet in waveform.
        rms_db (int): Reference value in dB to scale audio data.
        
    Returns:
        scaled_wav (torch.Tensor): Scaled wav tensor.
    """
    wav = torch.tensor(wav) if not isinstance(wav, torch.Tensor) else wav
    # rms value in dB to rescale the audios, convert from db to linear scale
    rms = 10**(rms_db / 20)
    # scaling factor per sample in batch (We want each slice to be on the same db value not the mean of all slices of a song)
    sc = torch.sqrt((wav.size(-1)*rms**2) / (torch.sum(torch.pow(wav, 2), dim=-1, keepdim=True)))
    return wav * sc


def peak_normalizer(wav: torch.Tensor) -> torch.Tensor:
    """Normalizes an audio signal by peak. Amplitudes of resulting signal are in the range [-1,1]."""
    if isinstance(wav, np.ndarray): wav = torch.tensor(wav)
    return wav / torch.abs(wav).max(dim=-1, keepdim=True)[0]


def adjust_vol(
    audio1: torch.Tensor | np.ndarray, 
    audio2: torch.Tensor | np.ndarray
) -> torch.Tensor:
    """Adjusts the volume of two audios. 

    By computing the amplitude ratio (by RMS) between the two signals, 
    the rms (loudness) of audio2 is aligned with the loudness of audio1.

    Args:
        audio1 (torch.Tensor): Audio waveform.
        audio2 (torch.Tensor): Audio waveform.
    
    Returns: 
        audio2_adjusted (torch.Tensor): Audio2 with loudness like audio1.
    """

    if isinstance(audio1, np.ndarray): audio1 = torch.tensor(audio1)
    if isinstance(audio2, np.ndarray): audio2 = torch.tensor(audio2)

    # defined function to compute rms of an audio
    def get_rms(sig: torch.Tensor):
        return torch.sqrt(torch.mean(torch.pow(sig, 2)))

    # compute amplitude ratio between the 2 udios that should be equalized in loudness
    amplitude_ratio = torch.abs(get_rms(audio1) / get_rms(audio2))
    # create instance of volume adjuster
    transform = torchaudio.transforms.Vol(gain=amplitude_ratio, gain_type="amplitude")
    # adjust volume and return
    return transform(audio2)


def normalize(mel: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Normalizes data batch in the range [-1,1]. 
    
    Data has to be in shape (num_channels, height, width) or (batch_size, num_channels, height, width).

    Args: 
        mel (torch.Tensor): Single mel-spectrorgam or batch of mels.
        epsilon (float): Summand to avoid division by zero.

    Returns:
        mel (torch.Tensor): Normalized mel or batch of mels.
    """
    mel_min = mel.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    mel_max = mel.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    # normalize each snippet (case 'valid' or 'train' we have several snippets (num_chunks))
    mel = 2*((mel - mel_min) / (mel_max - mel_min + epsilon)) - 1
    return mel
