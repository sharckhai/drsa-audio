import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import torchaudio
import librosa

from cxai.utils.visualization import plot_spectrogram, plot_loss_drsa
from cxai.utils.constants import CLASS_IDX_MAPPER



class Loader():

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
        """
        wav, _ = torchaudio.load(path_to_audio)
        wav = wav.requires_grad_(False)

        if self.case == 'gtzan':
            wav = get_slice(wav, slice_length=self.slice_length, num_chunks=num_chunks,
                            start_point=startpoint, sample_rate=self.sample_rate)
        
        # TODO: double normalization maybe change

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

        wav = peak_normalizer(wav)

        spec = self.wav2spec(wav)

        mel = self.spec2mel(torch.abs(spec))

        logmel = torch.log10(mel + 1e-7)

        if clamp: logmel = torch.clamp(logmel, -4)

        #logmel = normalize(logmel)

        if return_all:
            mag, phase = librosa.magphase(spec.numpy())
            return wav.numpy(), mag[..., :self.width], phase[..., :self.width], mel.numpy()[..., :self.width]
        
        if self.case == 'gtzan':
            return logmel[..., :self.width].reshape(-1,1,self.n_mels,self.width)
        else:
            return logmel[..., 1:-2].reshape(-1,1,self.n_mels,self.width)

    

    def transform_wav_(self, wav, return_all=False, clamp=True):

        wav = peak_normalizer(wav)

        spec = self.wav2spec(wav)

        mel = self.spec2mel(torch.abs(spec))

        logmel = torch.log10(mel + 1e-7)

        if clamp: logmel = torch.clamp(logmel, -4)

        #logmel = normalize(logmel)

        if return_all:
            mag, phase = librosa.magphase(spec.numpy())
            return wav.numpy(), mag[..., :self.width], phase[..., :self.width], mel.numpy()[..., :self.width]
        
        return logmel[..., :self.width]#.reshape(-1,1,128,self.width)
    


    def load_all_representations(self, path_to_audio, startpoint=None):
        return self.load(path_to_audio, startpoint, return_all=True)




# slice a piece of a specific wav
def get_slice(wav, slice_length: int = 6, start_point: int = 0, num_chunks: int = 1, sample_rate: int = 22050, return_startpoints=False):
    '''
    Takes a specific wav audio and slices it
    '''    
    window_size = int(slice_length * sample_rate)
    
    if num_chunks > 1:
        # we set minimum length of the audios to 29s (shortest audio is ~29.3s)
        hop = int(round_down(((29 - slice_length) / (num_chunks - 1)), 1) * sample_rate)
        # unfold along the time dimension
        wav_sample_sliced = wav[:, :29*sample_rate].unfold(1, window_size, hop).reshape(-1, 1, window_size)

        assert wav_sample_sliced.shape[0] == num_chunks, 'not equal num_chunks'
    else:
        start_sample = int(start_point * sample_rate)
        assert start_point <= wav.size(1) - window_size, f'Start_point has to be in range [{0},{wav.size(1) - window_size}]'
        wav_sample_sliced = wav[:, start_sample:start_sample+window_size]

    return wav_sample_sliced



def rms_normalizer(wav) -> torch.Tensor:
    
    # TODO: check This /20 or /10?
    # TODO: Which ref value?

    # rms value in dB to rescale the audios
    rms_db = 0
    rms = 10**(rms_db / 20)
    # scaling factor per sample in batch (We want each slice to be on the same db value not the mean of all slices of a song)
    sc = torch.sqrt((wav.size(-1)*rms**2) / (torch.sum(torch.pow(wav, 2), dim=-1, keepdim=True)))

    return wav * sc



def peak_normalizer(wav):
    return wav / torch.abs(wav).max(dim=-1, keepdim=True)[0]



# TODO: work on this
def adjust(sig1, sig2) -> torch.Tensor:
    print(torch.abs(sig1).max(), torch.abs(sig2).max())

    if torch.abs(sig1).max() < torch.abs(sig2).max():
        f = torch.abs(sig2).max() / torch.abs(sig1).max()
        f = f #- f/16
        clamp_value = torch.abs(sig1).max()
        sig2 = torch.clamp(sig2/f, -clamp_value, clamp_value) 
    else:
        clamp_value = torch.abs(sig2).max()
        sig1 = torch.clamp(sig1, -clamp_value, clamp_value) 
    
    # print(sig1rms.max(), sig1rms.min(), sig2rms.max(), sig2rms.min())

    return sig2 



def adjust_vol(orig_wav, subspace_audio) -> torch.Tensor:

    if isinstance(orig_wav, np.ndarray): orig_wav = torch.tensor(orig_wav)
    if isinstance(subspace_audio, np.ndarray): subspace_audio = torch.tensor(subspace_audio)

    def get_rms(sig):
        return torch.sqrt(torch.mean(torch.pow(sig, 2)))
    
    #print('RMS: ', get_rms(orig_wav), get_rms(subspace_audio))
    #print('PEAK: ', torch.abs(orig_wav).max(), torch.abs(subspace_audio).max())

    amplitude_ratio = torch.abs(get_rms(orig_wav) / get_rms(subspace_audio)) #// 2
    transform = torchaudio.transforms.Vol(gain=amplitude_ratio, gain_type="amplitude")
    subspace_audio_adjusted = transform(subspace_audio)
    #print('RMS: ', get_rms(orig_wav), get_rms(subspace_audio_adjusted))
    #print('PEAK: ', torch.abs(orig_wav).max(), torch.abs(subspace_audio_adjusted).max())
    return subspace_audio_adjusted



def normalize(mel, epsilon=1e-7) -> torch.Tensor:
    '''
    Normalizes e dimensional datapoints to the range [-1,1]
    Data has to be in shape (num_channels, height, width) or (batch_size, num_channels, height, width)
    '''

    mel_min = mel.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    mel_max = mel.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # normalize each snippet (case 'valid' or 'train' we have several snippets (num_chunks))
    mel = 2*((mel - mel_min) / (mel_max - mel_min + epsilon)) - 1
    
    return mel


def round_down(n, decimalpoints):
    return math.floor(n * 10**decimalpoints) / 10**decimalpoints




def load_sample(path_to_audio, slice_length: int=6, start_point: int=0, num_chunks: int=1, to_mel: bool=True, normalize=True, case='128_256', set='noisy', audio_normalization=None):
    '''
    Loads an audio from file, extracts a snippet from it and transforms it to mel-spectrogram
    (combines get_slice and transform_wav_to_mel)

    Parameters
    ----------
    path_to_audio: string
        path to audio file
    slice_length: int
        desired length of the audio snippet in seconds
    start_point: int
        point where to start the audio snippet in seconds
    to_mel: boolean
        arg if wav audio should be transformed to mel-specgtrogram in dB

    Returns
    -------
    sample: torch.Tensor
        Mel-spectrogram with amplitude scaled to dB
    genre: string
        genre of the audio sample
    '''
    # extract genre
    genre = str.split(path_to_audio, '/')[-2]

    # load wav
    waveform, _ = torchaudio.load(path_to_audio)
    waveform = waveform.requires_grad_(False)

    if case != 'toy':
        waveform = get_slice(waveform, slice_length, start_point, num_chunks)

    # normalize by rms 
    if audio_normalization == 'rms':
        waveform = rms_normalizer(waveform)
    elif audio_normalization == 'peak':
        waveform = peak_normalizer(waveform)
    else:
        pass

    if to_mel:
        if case == 'toy':
            # convert to mel-spectrogram
            mel = transform_wav_to_mel(waveform, sample_rate=16000, n_fft=480, n_mels=64, hop_length=240, normalize=normalize, case=case, set=set)
            # discard first and last 2 time bins to get shape 64x64
            mel = mel.unsqueeze(1)[..., 1:-2]
            return mel, genre
        elif case == '128_128':
            resampler = torchaudio.transforms.Resample(22050, 16000)
            waveform = resampler(waveform)
            mel = transform_wav_to_mel(waveform, sample_rate=16000, n_fft=800, n_mels=128, hop_length=360, normalize=normalize, case=case)
            mel = mel[..., 3:-3]
            return mel, genre
        elif case == '64_128':
            mel = transform_wav_to_mel(waveform, n_fft=2048, n_mels=64, hop_length=1024, normalize=normalize, case=case)
            mel = mel[..., 1:-1]
            return mel, genre
        elif case == '128_256':
            mel = transform_wav_to_mel(waveform, n_fft=1024, n_mels=128, hop_length=512, normalize=normalize, case=case, audio_normalization=audio_normalization)
            mel = mel[..., 1:-2]
            return mel, genre
        else:
            mel = transform_wav_to_mel(waveform, normalize=normalize, case=case)
            return mel, genre
    else:
        return waveform, genre



def transform_wav_to_mel(sliced_wav, sample_rate=22050, n_fft=1024, n_mels=128,  hop_length=512, normalize=True, case=None, set='baseline', audio_normalization='rms'):
    '''
    Takes an already sliced wav audio and transforms it to mel-spectrogramm with amplitude in dB

    Parameters
    ----------
    sliced_wav: torch.Tensor
        sliced audio snippet

    Returns
    -------
    mel_db: torch.Tensor
        Mel-spectrogram with amplitude scaled to dB
    '''

    assert type(sample_rate) == int, 'sample_rate must be an integer'
    assert type(n_fft) == int, 'n_ffth must be an integer'
    assert type(n_mels) == int, 'n_mels must be an integer'
    assert type(hop_length) == int, 'hop length must be an integer'

    # define pre calculated dataset statistics (calculated in data_preprocessing.py)
    if case=='toy':
        #if set == 'baseline':
        #    mean = 11.8361
        #    std = 8.6480
        #else:
        mean = -12.2433
        std = 11.7116
    elif case == 'square128':
        mean = 22.9381
        std = 14.7086
    elif case == '128_256':
        if audio_normalization=='rms':
            mean = 16.5831
            std = 15.0282
        else:
            mean = 1.1095
            std = 15.6003
    elif case == '64_128':
        mean = 27.4231
        std = 14.1295
    else:
        mean = 16.6152
        std = 15.0055

    # change to tensor for mel transformation
    if isinstance(sliced_wav, np.ndarray):
        sliced_wav = torch.tensor(sliced_wav)

    # to mel-spectrogram from wav
    mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, f_min=0.0, f_max=sample_rate/2, n_mels=n_mels, hop_length=hop_length)
    mel = mel_converter(sliced_wav)
    
    # amplitude to DB
    convert_to_dB = torchaudio.transforms.AmplitudeToDB()
    mel_db = convert_to_dB(mel)
    
    # normalize with pre calculated statistics
    if normalize:
        mel_db = normalize_mel(mel_db)
    else:
        mel_db = (mel_db - mean) / std
    
    return mel_db






### get suitable test sample for lrp
def get_toy_sample(dataset, class_, num, suffix='wav'):
    genre_idx = {"class1" : 0, "class2" : 1}
    
    # build path
    sample = f'/class{class_}/{num:05d}.{suffix}'
    path_to_sample = '/Users/samuelharck/Desktop/masterthesis/toydatasaw/' + dataset + sample
    #path_to_data_folders = os.path.join('/Users/samuelharck/Desktop/masterthesis/toydatasaw/', dataset, sample)
    #path_to_sample = os.path.join(path_to_data_folders, f'/class{class_}/{num:05d}.wav')

    # get wav
    wav, _ = torchaudio.load(path_to_sample)
    # load 5s long sample mel
    mel, genre = load_sample(path_to_sample, to_mel=True, case='toy', set=dataset)
    # create virtual batch dimension
    mel = mel.view(1, 1, 64, 64)

    # print class and mocel output
    print('Actual class: ', genre_idx[genre]+1)
    # plot mel
    plot_spectrogram(mel.squeeze())
    
    return wav, mel


def shuffle_and_truncate_databatch(data_batch, paths_to_songs, N, seed=42):

    # instantiate local generator
    local_gen = torch.Generator()
    local_gen = local_gen.manual_seed(seed)
    perm_mask = torch.randperm(data_batch.size(0), generator=local_gen)

    data_batch = data_batch[perm_mask][:N]
    paths_to_songs = [paths_to_songs[i] for i in perm_mask[:N]]

    return data_batch, paths_to_songs



###################################### functions to load paths to songs ######################################

def get_songlist_random(path, num_folds=5):

    songlist = []
    for fold in range(1,num_folds+1):
        list_filename = os.path.join(path, f'fold_{fold}.txt')
        with open(list_filename) as f:
            lines = f.readlines()
        # extend songlist
        songlist.extend([line.strip() for line in lines])

    return songlist


def get_songlist(path, 
                 genre: str = None,
                 excluded_folds: list = None,
                 num_folds: int = 10, 
                 return_list: bool = True, ##################### for what do we need this?
                 genres: dict = CLASS_IDX_MAPPER):
    """
    Function to get list of all files GTZAN dataset.

    Args:
    _____
    folds: list
        list of fold from which the songs shold be loaded. If None, all songs from all folds are loaded.
    return_dict: bool
        defines if dictionary of genres with a list of songs shold be returned or just a big list with all sings of every genre
    """

    assert not path.startswith('.'), 'path has to be an absolute path'

    genres = [genre] if genre else genres

    songdict = {}
    songlist = []

    for key in genres:
        songs_of_genre = get_songs_of_genre(path, key, excluded_folds, num_folds)

        if return_list:
            songlist.extend(songs_of_genre)
        else:
            songdict[key] = songs_of_genre

    songs = songlist if return_list else songdict
    return songs

# TODO: if no genre is provided just load and concat all folds



def get_songs_of_genre(path, genre, excluded_folds: list = None, num_folds: int = 10):
    """
    Returns list with paths to all samples of the specified genre
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




def get_toy_samplelist(path, toyclass=None, splits: str=None):
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




def plot_runs(path, runs=1):
    stats = []
    if runs == 1:
        stats.append(get_train_stats(path))
    else:
        for i in range(1,runs+1):
            stats.append(get_train_stats(os.path.join(path, f'run{i}')))
    plot_loss_drsa(stats)
    return stats



def get_best_run(path):

    best_loss = 0
    concept_relevances = None

    for dir_level1 in [d for d in os.listdir(path) if not d.startswith('.')]:
        run = int(dir_level1[-1])
        loss, concept_relevances, train_losses = get_run_stats(os.path.join(path, dir_level1, 'train_stats.csv'))

        if loss > best_loss:
            best_loss = loss
            concept_relevances = concept_relevances
            best_run = run
            path_to_best_run = os.path.join(path, dir_level1)

    return best_run, best_loss, concept_relevances, path_to_best_run, train_losses

def get_run_stats(path):

    stats = pd.read_csv(path)
    final_loss = list(stats['loss'])[-1]

    concept_relevances = []
    for key in stats.keys():
        if key.startswith('R'):
            concept_relevances.append(list(stats[key])[-1])
    
    return final_loss, concept_relevances, list(stats['loss'])

'''def get_song_list_filtered(path_to_txt, split, genre):

    splits = ['train', 'valid', 'test'] if split=='filtered' else [split.split('_')[0]]

    paths = []

    for split in splits:
        print(split)

        list_filename = os.path.join(path_to_txt, '%s_filtered.txt' % split)
        with open(list_filename) as f:
            lines = f.readlines()

        for line in lines:
            if line.split('/')[0] == genre and line.split('/')[1].split('.')[1] != '00054':
                paths.append(os.path.join(path_to_txt, 'genres_original', line.strip()))

    return paths
'''



###################################### fucntions for accuracy model evaluation ######################################

def get_acc(model, testloader=None, device=torch.device('cpu'), is_toy=False):
    '''
    Calculates accuracy on test set

    Parameters
    ----------
    model: nn.Sequential
        ML model
    testloader: torch.loader
        testloader
    device: torch.device()
        torch device to perform calculations on
    is_toy: bool
        Case toy model and toydataset

    Returns
    -------
    acc: float
        mean accuracy on all classes
    ypred: list
        predicted labels on test set
    ytrue: list
        true labels of test set
    '''

    model.eval()

    ytrue = []
    ypred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in testloader:
            if not is_toy:
                b, chunks, c, f, t = xb.size() # b=batch
                xb = xb.view(-1, c, f, t)
                yb = yb.repeat_interleave(chunks)

            xb, yb = xb.to(device).squeeze(0), yb.to(device)
            #yb = yb.repeat_interleave(xb.size(0))

            outputs = model(xb)
            _, predicted = torch.max(outputs.detach(), 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            ytrue.extend(yb.cpu().numpy())
            ypred.extend(predicted.cpu().numpy())

    acc = correct / total * 100
    print(f'Accuracy on test set is: {round(acc, 2)}%')

    return acc, np.asarray(ytrue), np.asarray(ypred)


def get_cm(ytrue, ypred, valid_fold=1, plot=True, genres=CLASS_IDX_MAPPER):
    '''
    Calculates and plots confusion matrix.

    Parameters
    ----------
    ypred: list
        predicted labels on test set
    ytrue: list
        true labels of test set
    plot: bool
        Controls if confusion matrix has to be plotted

    Returns
    -------
    cm: np.array() 
        confusion matrix of shape [num_classes, num_classes]
    '''

    cm = confusion_matrix(ytrue, ypred)
    # convert to percentage
    cm = cm / cm.sum(axis=1) * 100

    if plot:
        plot_cm(cm, valid_fold=valid_fold)
    
    return cm

def plot_cm(cm, valid_fold=None):
    ax = sns.heatmap(cm, annot=True, xticklabels=CLASS_IDX_MAPPER.keys(), yticklabels=CLASS_IDX_MAPPER.keys(), cmap='YlGnBu', fmt='.1f')

    # Setting the title, x-axis label, and y-axis label
    ax.set_title('Confusion Matrix across 10 folds [%]' if valid_fold==None else f'Confusion Matrix [%], Validation fold: {valid_fold}')  # Main title
    ax.set_xlabel('Predicted label')  # X-axis label
    ax.set_ylabel('True label')  # Y-axis label
    plt.show()


def class_accs(cm, genres=CLASS_IDX_MAPPER):
    '''
    Calculates and prints prediction accuaries of each class.

    Parameters
    ----------
    cm: np.array()
        confusion matrix

    Returns
    -------
    confusion_matrix_dict: dict
        prediction accuracy on each class 
    '''

    confusion_matrix_dict = {}
    class_accuracies = np.diag(cm) / np.sum(cm, axis=1) * 100
    for i, class_acc in enumerate(class_accuracies):
        print(f'Acc of {list(genres.keys())[i]:>10s}: {round(class_acc, 2):.2f}%')
        confusion_matrix_dict[list(genres.keys())[i]] = round(class_acc, 2)
    return confusion_matrix_dict



###################################### fucntions for training and model evaluation ######################################

def get_train_stats(path, epoch=None, old=None):
    
    if path.endswith('.csv'):
        csv_files = [path]
    else:
        # case training was interrrupted and carried on at some later point (several .csv files for train stats)
        csv_files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.csv')]
        csv_files.sort()
    
    all_dfs = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        all_dfs.append(df[['train_loss', 'train_acc', 'valid_losses', 'valid_acc']])

    return pd.concat(all_dfs, ignore_index=True)



class HiddenPrints:
    """
    Class to hide prints from 'fit_baseline()' function during grid search.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout