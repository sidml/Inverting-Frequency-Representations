import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot_val_loss(all_val_loss):    
    plt.figure(figsize=(10, 5))
    plt.plot(all_val_loss)
    plt.title('Validation Loss vs Epoch')
    plt.xlabel('Epoch')

def compare_wave(pred, actual):
    plt.figure(figsize=(20, 5))
    plt.plot(pred, alpha=0.5)
    plt.plot(actual, alpha=0.5)
    plt.legend(['pred', 'actual'])
    plt.title('Predicted vs Actual')


def compare_stft(orig_stft, all_pred_stft):
    a = orig_stft.reshape(-1, 2050)
    b = all_pred_stft.reshape(-1, 2050)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(a[:, :1025].mean(0), alpha=0.5)
    plt.plot(b[:, :1025].mean(0), alpha=0.5)
    plt.legend(['original mag', 'predicted mag'])
    plt.title('Magnitude Comparision')

    plt.subplot(1, 2, 2)
    plt.plot(a[:, 1025:].mean(0), alpha=0.5)
    plt.plot(b[:, 1025:].mean(0), alpha=0.5)
    plt.legend(['original angle', 'predicted angle'])
    plt.title('Angle Comparision')


def stft_plotter(stft, y, sr=22050, time=[2, 3]):
    S = np.abs(stft)
    for t in time:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))

        img = librosa.display.specshow(librosa.amplitude_to_db(S[:, t:t+1],
                                                               ref=np.max),
                                       y_axis='log', x_axis='time', ax=ax[0])

        ax[0].set_title('Power spectrogram')

        img = librosa.display.specshow(librosa.amplitude_to_db(np.real(stft[:, t:t+1]),
                                                               ref=np.max),
                                       y_axis='log', x_axis='time', ax=ax[1])

        ax[1].set_title('Real Part of STFT')

        img = librosa.display.specshow(librosa.amplitude_to_db(np.imag(stft[:, t:t+1]),
                                                               ref=np.max),
                                       y_axis='log', x_axis='time', ax=ax[2])

        ax[2].set_title('Imaginary Part of STFT')

        librosa.display.waveplot(y[sr*t:sr*(t+1)], sr=sr, ax=ax[3])
        ax[3].set_title('Waveform')
