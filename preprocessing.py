import numpy as np
import mne
import torch 
import torch.fft as fft
import julius
from scipy import signal

#%%
# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 100
    else:
        resampling_frequency = 100
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = signal.resample_poly(data, up, down, axis=1)

    return data, resampling_frequency


#%%
def segment_eeg_signal(eeg_signal, window_size, step_size, Fs):
    """
    window_size ---> in min
    window_step ---> in min
    """
    if Fs == 0:
        Fs = 100

    # Calculate the number of samples per window
    window_samples = int(round(window_size*60*Fs))

    # Calculate the number of samples to move the window by
    overlap_samples = int(step_size*60*Fs)

    # Determine the total number of windows
    num_windows = int(np.ceil((eeg_signal.shape[1] - window_samples) / overlap_samples)) + 1

    # Segment the EEG signal into windows
    segments = []
    for i in range(num_windows):
        start_index = i * overlap_samples
        end_index = start_index + window_samples
        
        if end_index > eeg_signal.shape[1]:
            break
        window = eeg_signal[:, start_index:end_index]
        segments.append(window)
    return segments


#%%
def resampling(config, data, sampling_frequency):

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = config.resampling_frequency
    else:
        resampling_frequency = config.resampling_frequency

    data_resampled = julius.resample_frac(data, int(sampling_frequency), int(resampling_frequency))

    return data_resampled, resampling_frequency

def bandpassing(data, sampling_frequency, device):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    low = passband[0]/int(sampling_frequency)
    high = passband[1]/int(sampling_frequency)

    bandpass = julius.BandPassFilter(low, high)
    bandpass = bandpass.to(device)
    data = bandpass(data)
    
    return data

def bandpassing_fft(config,data, sampling_frequency, device):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]


    # Calculate the FFT of the EEG-like signal
    eeg_fft = fft.fft(data)

    # Define the bandpass filter in the frequency domain
    nyquist_freq = sampling_frequency/ 2
    num_samples =  len(eeg_fft[1])

    # Calculate the frequency values corresponding to the FFT components on the CPU
    freqs = np.fft.fftfreq(num_samples, 1.0 / sampling_frequency)
    freqs = torch.tensor(freqs, device=device)

    mask = (freqs >= passband[0]) & (freqs <= passband[1])

    # Apply the filter

    eeg_fft_filtered = eeg_fft * mask

    # Calculate the inverse FFT to convert the filtered signal back to the time domain
    eeg_filtered = fft.ifft(eeg_fft_filtered)

    #print(eeg_fft_filtered)
    return  eeg_filtered.real
#%%
def rescale_data(data):
    # Scale the data to the interval [-1, 1].
    min_value = torch.min(data)
    max_value = torch.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data
    return data 