import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa

from zennit.image import imgify
from zennit.cmap import ColorMap



# DRSA subplots

def make_drsa_subplot(mel, standard_heatmap, subspace_heatmaps, cmap=None, case=None, figsize=(14, 7)):#, scaling_factor=1):
    # Create figure
    fig = plt.figure(figsize=figsize)

    # Define the grid layout
    gs = gridspec.GridSpec(2, 4)
    #fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(18, 3))
    cmap = ColorMap('008,2:00f,4:00f,80:fff,b:f00,d:f00,800') if cmap is None else cmap

    # First row
    ax1 = fig.add_subplot(gs[0, 1])  # Spans first two columns of first row
    ax2 = fig.add_subplot(gs[0, 2])  # Spans last two columns of first row

    #ax1.imshow(mel.squeeze(), origin="lower", aspect='equal')
    #ax1.show(plot_spectrogram(mel, show=False))
    #ax1.set_title('Mel-Spectrogram')
    #ax1.axis('off')
    
    # show ax 1
    plot_spectrogram(mel, ax1, case=case, colorbar=False)

    # normalize to [0,1] and map 0 relevance to 0.5 (case: symmetric)
    vmax = standard_heatmap.max()
    vmin = -vmax

    # Set titles for demonstration
    ax2.imshow(vis_heatmap(standard_heatmap, vmin, vmax, scaling_factor=1, cmap=cmap, case=case))
    ax2.set_title(r'$\sum_{i,k} R_{ik}$' + f'={standard_heatmap.sum():5.2f}')
    # delete axis ticks, only keep black frame
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Example plotting commands for the right plots
    for i in range(len(subspace_heatmaps)):#, ax in enumerate([ax3, ax4, ax5, ax6]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(vis_heatmap(subspace_heatmaps[i], vmin, vmax, scaling_factor=1, cmap=cmap, case=case))
        ax.set_title(r'$\sum_i R_{i,k}$' + f'={subspace_heatmaps[i].sum():5.2f}')
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, top=0.8)
    plt.figtext(0.615,0.88,"Standard Heatmap", va="center", ha="center", size=13)
    plt.figtext(0.5,0.45,"Subspace Heatmaps", va="center", ha="center", size=13)
    plt.show()

# DRSA subplots

def make_drsa_subplot_8(mel, standard_heatmap, subspace_heatmaps, subspace_relevances, cmap=None, case=None, figsize=(16,12)):#, scaling_factor=1):
    # Create figure
    fig = plt.figure(figsize=figsize)

    # Define the grid layout
    gs = gridspec.GridSpec(3, 4)
    
    #fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(18, 3))
    cmap = ColorMap('008,2:00f,4:00f,80:fff,b:f00,d:f00,800') if cmap is None else cmap

    # First row
    ax1 = fig.add_subplot(gs[0, 1])  # Spans first two columns of first row
    ax2 = fig.add_subplot(gs[0, 2])  # Spans last two columns of first row
    # show ax 1
    #ax1.imshow(plot_spectrogram(mel, case=case))
    #ax1.axis('off')
    plot_spectrogram(mel, ax=ax1, case=case, colorbar=False)

    # normalize to [0,1] and map 0 relevance to 0.5 (case: symmetric)
    vmax = standard_heatmap.max()
    vmin = -vmax

    # Set titles for demonstration
    ax2.imshow(vis_heatmap(standard_heatmap, vmin, vmax, scaling_factor=1, cmap=cmap, case=case))
    ax2.set_title(r'$\sum_{i,k} R_{ik}$' + f'={standard_heatmap.sum():5.2f}')
    # delete axis ticks, only keep black frame
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Example plotting commands for the right plots
    for i in range(len(subspace_heatmaps)):#, ax in enumerate([ax3, ax4, ax5, ax6]):
        ax = fig.add_subplot(gs[(i//4)+1, i-(i//4)*4])
        ax.imshow(vis_heatmap(subspace_heatmaps[i], vmin, vmax, scaling_factor=1, cmap=cmap, case=case))
        ax.set_title(r'$\sum_i R_{i,k}$' + f'={subspace_relevances[i]:5.2f}')
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout
    plt.subplots_adjust(hspace=0.6, top=0.8)
    plt.figtext(0.61,0.85,"Standard Heatmap", va="center", ha="center", size=14)
    plt.figtext(0.5,0.33,"Subspace Heatmaps", va="center", ha="center", size=14)
    plt.show()



def make_drsa_subplot_2(mel, standard_heatmap, subspace_heatmaps, subspace_relevances, cmap=None, case=None, figsize=(17,4)):
    # Create figure
    fig = plt.figure(figsize=figsize)

    # Define the grid layout
    gs = gridspec.GridSpec(1, 4)
    #fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(18, 3))
    cmap = ColorMap('008,2:00f,4:00f,80:fff,b:f00,d:f00,800') if cmap is None else cmap
    cmap = ColorMap('00f,80:fff,9:f00,d:000')

    # First row
    ax1 = fig.add_subplot(gs[0, 0])  # Spans first two columns of first row
    ax2 = fig.add_subplot(gs[0, 1])  # Spans last two columns of first row
    # show ax 1
    plot_spectrogram(mel, ax1, case=case, n_c=2)

    # normalize to [0,1] and map 0 relevance to 0.5 (case: symmetric)
    vmax = standard_heatmap.max()
    vmin = -vmax

    # Set titles for demonstration
    ax2.imshow(vis_heatmap(standard_heatmap, vmin, vmax, scaling_factor=1, cmap=cmap, case=case))
    ax2.set_title(r'$\sum_{i,k} R_{ik}$' + f'={standard_heatmap.sum():5.2f}')
    # delete axis ticks, only keep black frame
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Example plotting commands for the right plots
    for i in range(len(subspace_heatmaps)):#, ax in enumerate([ax3, ax4, ax5, ax6]):
        ax = fig.add_subplot(gs[0, i+2])
        ax.imshow(vis_heatmap(subspace_heatmaps[i], vmin, vmax, scaling_factor=1, cmap=cmap, case=case))
        ax.set_title(r'$\sum_i R_{i,k}$' + f'={subspace_relevances[i]:5.2f}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(top=0.7)
    plt.figtext(0.41,0.85,"Standard Heatmap", va="center", ha="center", size=15)
    plt.figtext(0.71,0.85,"Subspace Heatmaps", va="center", ha="center", size=15)
    plt.show()


def vis_heatmap(relevance, vmin=None, vmax=None, scaling_factor=3, cmap=None, case=None):

    relevance = relevance.detach().cpu().numpy() if type(relevance) == torch.Tensor else relevance

    # mirror the rows of relevances because mel-spectrograms contain first mel bin as first row (so highest frequency is in lowest row)
    # when plotting one sets plt.imshow(ORIGIN='LOWER'), so first row is displayed in lowest row of the plot
    relevance = np.flip(relevance, axis=[-2])

    cmap = ColorMap('00f,80:fff,9:f00,d:000') #if cmap is None else cmap

    if case == 'toy':
        # we use zplus and an usgined colormap
        #vmin = 0 
        # create an image of the visualize attributio
        #relevance = np.clip(relevance, a_min=0, a_max=None)
        #cmap = ColorMap('008,2:00f,4:00f,80:fff,b:f00,d:f00,800') if cmap is None else cmap
        img = imgify(relevance.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, symmetric=False)
    else:
        #cmap = ColorMap('008,2:00f,4:00f,80:fff,b:f00,d:f00,800') if cmap is None else cmap
        # create an image of the visualize attribution
        img = imgify(relevance.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, symmetric=True)
        #img = imgify(relevance.squeeze(), cmap='bwr', vmin=vmin, vmax=vmax, symmetric=True)

    # resize image
    new_size = (img.size[0]*scaling_factor, img.size[1]*scaling_factor)

    return img.resize(new_size).convert('RGB')


def plot_spectrogram(mel, ax=None, sr=16000, case=None, n_c=4, colorbar=True):
    '''
    Plots spectrogram

    Parameters
    ----------
    spectrogram: torch.Tensor
        spectrogram in any form (mel-spectrogram, dB, ...)
    sr: int
        sample rate
    title: string
        title for plot
    '''

    mel = mel.squeeze()

    # to numpy for pyplot
    if not isinstance(mel, np.ndarray):
        mel = mel.numpy()

    # define frequencies to plot on y axis
    frequencies = [512, 1024, 2048, 4096]
    # convert freqs to mel bins
    mel_bins = librosa.hz_to_mel(frequencies)

    
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(label='Mel-Spectrogram')
    else:
        # adjust ittle for drsa subplot
        if n_c == 2:
            ax.set_title(label='Mel-Spectrogram', y=1.12, size=15)
        elif n_c == 4:
            ax.set_title(label='Mel-Spectrogram', y=1.12, size=13)

    # Create the plot
    img = librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=sr, hop_length=240 if case == 'toy' else 360, ax=ax, htk=True, cmap='viridis')
    if colorbar==True:
        ax.figure.colorbar(img, ax=ax, format='%+2.0f ' + r'$\log_{10}(A)$')

    # set y-ticks to the desired frequencies
    ax.set_yticks(librosa.mel_to_hz(mel_bins))
    ax.set_yticklabels([f'{freq} Hz' for freq in frequencies])

    if case=='toy':
        ax.set_xticks([0, mel.shape[-1]/2/68, mel.shape[-1]/68])
        ax.set_xticklabels(['0', '0.5', '1'])
    else:
        ax.set_xticks(np.arange(6)/(2))

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Mel bins')

    if case == 'toy':
        ax.set_xlim(0, 0.942)  # Extend the x-axis limit
        ax.set_ylim(0, 8000) 
        ax.set_aspect(1 / 5700)
    else:
        #ax.set_xlim(0, 3)  # Extend the x-axis limit
        ax.set_ylim(0, 8000) 
        ax.set_aspect(1 / 1800)

    #ax.set_title(label='Mel-Spectrogram', y=1.06)


# AUDIO visulaization

def plot_waveform(wav, sample_rate=16000):
    '''
    Plots waveform audio

    Parameters
    ----------
    waveform: torch.Tensor
        audio wav
    sample_rate: int
        sample rate of the provided audio
    '''
    # tranform to numpy and extract shape if data
    if not isinstance(wav, np.ndarray):
        wav = wav.numpy()
    num_channels, num_frames = wav.shape

    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, wav[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("Waveform")
    #axes.set_xlabel('Amplitude normalized')
    #axes.set_ylabel('time in s')
    plt.show(block=False)



def plot_acc(df, compare_models=False, smoothed=False, label='None', legend=True, valid_key = 'valid_acc'):
    if smoothed == 1:
        if compare_models:
            plt.plot(df['valid_acc_smoothed'], label=label)
        else:
            plt.plot(df['valid_acc_smoothed'], label='Sm. Val. Acc', color='green')
    elif smoothed == 2:
        plt.plot(df['valid_acc_smoothed'], label='Sm. Val. Acc', color='green')
        plt.plot(df['validation_accuracy'], label='Valid Acc', color='orange')
    elif smoothed == 3:
        plt.plot(df['valid_acc_smoothed'], label='Sm. Val. Acc', color='green')
        plt.plot(df['train_acc_smoothed'], label='Sm. Tr. Acc', color='blue')
    elif smoothed == 0:
        plt.plot(df[valid_key], label='Valid Acc', color='orange')

    # plot train accuracy and stats only if consideruing one model
    if not compare_models:
        plt.plot(df['train_acc'], label='Train Acc', color='blue')
        plt.axhline(y=max(df[valid_key]), color='r', linestyle='-')
        print('%15s: %6.2f%% (epoch: %4d)' % ('Best accuracy', (max(df[valid_key])*100), np.argmax(df[valid_key])))
        print('%15s: %6.2f%% (epoch: %4d)' % ('Final accuracy', (list(df[valid_key])[-1] * 100), len(df)))

    plt.legend() if legend else None
    plt.title('Acc over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    


def plot_loss(df, train_key='train_loss', valid_key='valid_losses'):
    plt.plot(df[train_key], label='Train Loss', color='blue')
    plt.plot(df[valid_key], label='Valid Loss', color='orange')
    plt.axhline(y=min(df[valid_key]), color='r', linestyle='-')
    plt.legend()
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # print min loss with respective epoch and final loss
    print('%15s: %6.4f (epoch: %4d)' % ('Min loss', min(df[valid_key]), np.argmin(df[valid_key])))
    print('%15s: %6.4f (epoch: %4d)' % ('Final loss', list(df[valid_key])[-1], len(df)))
        

def plot_loss_drsa(stats):

    for i, df in enumerate(stats, start=1):
        plt.plot(df['loss'], label=f'run{i}')
        #plt.axhline(y=max(df['loss']), color='r', linestyle='-')
        plt.legend()
        plt.title('Loss during run')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        # print min loss with respective epoch and final loss
        print('RUN%1d:\n%10s: %6.4f (epoch: %4d)' % (i, 'Final loss', list(df['loss'])[-1], len(df)))
        print('-'*5)


def plot_concept_relevances_drsa(df):

    mean_ = np.zeros(len(df))
    num_concepts = 0

    for key in df.keys():
        if key.startswith('R'):
            mean_ += np.array(df[key])
            num_concepts += 1
            plt.plot(df[key], label=key)
            plt.legend()
            plt.title('Concept relevances during run')
            plt.xlabel('Iteration')
            plt.ylabel('R')
            # print min loss with respective epoch and final loss
            print('Concept%1d:\n%10s: %6.4f (epoch: %4d)' % (int(key[-1]), 'Final R', list(df[key])[-1], len(df)))
            print('-'*5)
    #plt.plot(mean_/num_concepts, label='mean relevance', color='lime')
    #plt.legend()
            



def plot_waveform(wav, sample_rate=22050):
    '''
    Plots waveform audio

    Parameters
    ----------
    waveform: torch.Tensor
        audio wav
    sample_rate: int
        sample rate of the provided audio
    '''
    # tranform to numpy and extract shape if data
    if not isinstance(wav, np.ndarray):
        wav = wav.numpy()
    num_channels, num_frames = wav.shape

    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, wav[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("Waveform")
    #axes.set_xlabel('Amplitude normalized')
    #axes.set_ylabel('time in s')
    plt.show(block=False)



def plot_aupcs(aupc_scores, averaged_pertubed_prediction_logits, flips_per_perturbation_step, title='EpsGammaWSquare'):

    for key in aupc_scores:
        # get data for each configuration 
        x_flipped_patches = np.cumsum(np.array(flips_per_perturbation_step)) / np.array(flips_per_perturbation_step).sum() * 100
        y_prediction_logits = np.array(averaged_pertubed_prediction_logits[key])#.flatten()

        if key[:5] == 'alpha':
            label = r'$\alpha$' + f' = {float(key[6:9]):3.1f}, ' + r'$\beta$' + ' = %3.1f, AUPC: %.3f' % (float(key[6:9])-1, aupc_scores[key].mean())
        elif key[:5] == 'zplus':
            label = 'zplus, AUPC: %.3f' % (aupc_scores[key].mean())
        else:
            if float(str.split(key, '_')[1]) > 1: continue
            label = r'$\gamma$' + ' = %3.2f, AUPC: %.3f' % (float(str.split(key, '_')[1]), aupc_scores[key].mean())
        #plt.plot(x_flipped_patches, y_prediction_logits, label='%15s AUPC: %6.2f' % (key, self.aupc_scores[key]), marker='o')
        plt.plot(x_flipped_patches, y_prediction_logits, label=label, marker='o')
        plt.title(f'AUPC Curves {title}')
        plt.xlabel('Flipped patches [%]')
        plt.ylabel('Averaged target class logit')
        plt.grid(ls=':', alpha=0.5)
        plt.legend()



def plot_spectrogram_old(spectrogram, sr=22050, title=None, ylabel=None, aspect="auto", to_freq=False, db_scale=True):
    '''
    Plots spectrogram

    Parameters
    ----------
    spectrogram: torch.Tensor
        spectrogram in any form (mel-spectrogram, dB, ...)
    sr: int
        sample rate
    title: string
        title for plot
    ylabel: string
        label for y axis
    aspect: string
        aspect ratio
    to_freq: bool
        change yscale to frequency. ONLY VALID FOR SPECTROGRAMS (not mel-specs, etc)
    '''

    # to numpy for pyplot
    if not isinstance(spectrogram, np.ndarray):
        spectrogram = spectrogram.numpy()

    spectrogram = spectrogram.squeeze()

    fig = plt.figure()
    plt.title(title if title else "Mel-Spectrogram")
    plt.ylabel(ylabel if ylabel else'Mel bins')
    plt.xlabel("Time bins")
    im = plt.imshow(spectrogram, origin="lower", aspect='auto')
    fig.colorbar(im, format='%+2.0f dB' if db_scale else '%+2.0f')
    plt.show(block=False)

