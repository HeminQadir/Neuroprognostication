import numpy as np
import os
import mne
import torch 
import torch.fft as fft
import julius
from scipy import signal
from helper_code import *


#%%
# Preprocess data.
def preprocess_data(data, sampling_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    # if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
    # data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs= 'cuda', verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs= 'cuda', verbose='error')
   
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

def load_recording_header(record_name, check_values=True):
    # Allow either the record name or the header filename.
    root, ext = os.path.splitext(record_name)
    if ext=='':
        header_file = record_name + '.hea'
    else:
        header_file = record_name

    # Load the header file.
    if not os.path.isfile(header_file):
        raise FileNotFoundError('{} recording not found.'.format(record_name))

    with open(header_file, 'r') as f:
        header = [l.strip() for l in f.readlines() if l.strip()]

    # Parse the header file.
    record_name = None
    sampling_frequency = None
    length = None 

    for i, l in enumerate(header):
        arrs = [arr.strip() for arr in l.split(' ')]
        # Parse the record line.
        if i==0:
            record_name = arrs[0]
            sampling_frequency = float(arrs[2])
            length = int(arrs[3])

    return sampling_frequency, length

def processrecord(config,recording_location,eeg_channels,device):
            
    if os.path.exists(recording_location + '.hea'):
       sampling_frequency, length = load_recording_header(recording_location, check_values=True)  # we created to read only the header and get the fs
     
       five_min_recording = sampling_frequency * 60 * config.window_size

                # checking the length of the hour recording 
       if length >= five_min_recording:
          data, channels, sampling_frequency = load_recording_data(recording_location, check_values=True)
          data = torch.tensor(data, dtype=torch.float32)
          data = data.to(device)

                    # checking if we have all the channels 
          if all(channel in channels for channel in eeg_channels):
             data, channels = reduce_channels(data, channels, eeg_channels)
             data, resampling_frequency = resampling(config, data, sampling_frequency)
                    
                        #start_time = time.time()
                   

             data = bandpassing(data, resampling_frequency , device)

                        #end_time = time.time()
                        #elapsed_time = end_time - start_time
                        #print(f"Elapsed time: {elapsed_time} seconds")

                      
                        
             bipolar_data = torch.zeros((config.in_channels, data.shape[1]), dtype=torch.float32)
             bipolar_data = bipolar_data.to(device)
             bipolar_data[0,:] = data[0,:] - data[1,:]   # Convert to bipolar montage: F3-P3 and F4-P4 
             bipolar_data[1,:] = data[2,:] - data[3,:]

                        #data = rescale_data(data)
             for k in range(0, config.in_channels):
                bipolar_data[k,:] = rescale_data(bipolar_data[k,:])
                            # print(bipolar_data[k,:].max())
                            # print(bipolar_data[k,:].min())
                            
             else:
                        pass
          else:
                    pass
    else: 
                pass
    return bipolar_data
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

    bandpass = julius.BandPassFilter(low, high,fft= False)
    bandpass = bandpass.to(device)
    data = bandpass(data)
    
    return data

def bandpassing_fft(config,data, sampling_frequency, device):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    print('i am in bandpass')
    # Calculate the FFT of the EEG-like signal
    eeg_fft = fft.fft(data)
    print(eeg_fft.shape)
    # Define the bandpass filter in the frequency domain
    #nyquist_freq = sampling_frequency/ 2
    num_samples =  len(eeg_fft[1])
    print( num_samples)

    # Calculate the frequency values corresponding to the FFT components on the CPU
    freqs = np.fft.fftfreq(num_samples, 1.0 / sampling_frequency)
    freqs = torch.tensor(freqs, device=device)
    print(freqs.shape)
    mask = (freqs >= passband[0]) & (freqs <= passband[1])
    print(mask.shape)
    # Apply the filter

    eeg_fft_filtered = eeg_fft * mask
    print(eeg_fft_filtered.shape)

    # Calculate the inverse FFT to convert the filtered signal back to the time domain
    eeg_filtered = fft.ifft(eeg_fft_filtered)
    print(eeg_filtered.shape)
    eeg_filtered_real= eeg_filtered.real
    print(eeg_filtered_real.shape)
    print('end of bandpass')
    return  eeg_filtered_real
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