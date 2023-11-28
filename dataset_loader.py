from helper_code import *
import torch
import random
from preprocessing import * 
from torch.utils.data import Dataset
import time


#%%
# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm), dtype=np.float32)

    return features

#%%
# Load the WFDB data for the Challenge (but not all possible WFDB files).
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

#%%
def number_to_one_hot(number, num_classes):
    identity_matrix = torch.eye(num_classes)
    one_hot = identity_matrix[number]
    return one_hot

#%%
def get_labels(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    # Extract labels.
    outcome = int(get_outcome(patient_metadata))

    outcome = torch.tensor(outcome, dtype=torch.long)
    return outcome

#%%
def load_data(config, data_folder, patient_id, device, train=True):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    #patient_features = get_patient_features(patient_metadata)
    # Handle NaN value
    #patient_features[patient_features != patient_features] = 0.0
    #print(patient_features)

    # Load EEG recording.    
    eeg_channels = ['T3', 'T4', 'T5', 'T6', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2']  #['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2'] # #['F3', 'P3', 'F4', 'P4'] 
    S            = [' 0', ' 1', ' 2', ' 3', ' 4', ' 5', '  6', '  7', ' 8', ' 9']
    group = 'EEG'

    
    resampling_frequency = config.resampling_frequency 
    window_size = config.window_size
    step_size   = config.step_size
    size = config.resampling_frequency * config.window_size * 60
    bipolar_data = torch.zeros((config.in_channels, size), dtype=torch.float32)
    bipolar_data = bipolar_data.to(device)

    # check if there is at least one EEG record
    if num_recordings > 0:
        random.shuffle(recording_ids)

        for recording_id in recording_ids:    #for recording_id in reversed(recording_ids):
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
            if os.path.exists(recording_location + '.hea'):
                sampling_frequency, length = load_recording_header(recording_location, check_values=True)  # we created to read only the header and get the fs
                five_min_recording = sampling_frequency * 60 * window_size

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
                        #data , resampling_frequency = preprocess_data(data, sampling_frequency)
                        #data = bandpassing_fft(config,data, sampling_frequency, device)

                        data = bandpassing(data, resampling_frequency , device)

                        #end_time = time.time()
                        #elapsed_time = end_time - start_time
                        #print(f"Elapsed time: {elapsed_time} seconds")

                        #data = torch.tensor(data, dtype=torch.float32)
                        #data = data.to(device)
                        #data, resampling_frequency = resampling(config, data, sampling_frequency)
                        
                        bipolar_data = torch.zeros((config.in_channels, data.shape[1]), dtype=torch.float32)
                        bipolar_data = bipolar_data.to(device)
                        #bipolar_data[0,:] = data[0,:] - data[1,:]   # Convert to bipolar montage: F3-P3 and F4-P4 
                        #bipolar_data[1,:] = data[2,:] - data[3,:]
                        bipolar_data[0,:] = data[0,:] - data[1,:]    # T3 - T4
                        bipolar_data[1,:] = data[2,:] - data[3,:]    # T5 - T6
                        bipolar_data[2,:] = data[4,:] - data[5,:]    # F7 - F8
                        bipolar_data[3,:] = data[6,:] - data[7,:]    # Fp1 - Fp2
                        bipolar_data[4,:] = data[0,:] - data[8,:]    # T3 - O1
                        bipolar_data[5,:] = data[1,:] - data[9,:]    # T4 - O2 
                        bipolar_data[6,:] = data[4,:] - data[8,:]    # F7 - O1
                        bipolar_data[7,:] = data[5,:] - data[9,:]    # F8 - O2
                        bipolar_data[8,:] = data[6,:] - data[8,:]    # Fp1 - O1
                        bipolar_data[9,:] = data[7,:] - data[9,:]    # Fp2 - O2

                        #data = rescale_data(data)
                        for k in range(0, config.in_channels):
                            bipolar_data[k,:] = rescale_data(bipolar_data[k,:])
                            # print(bipolar_data[k,:].max())
                            # print(bipolar_data[k,:].min())
 

                        break
                    else:
                        pass
                else:
                    pass
            else: 
                pass
    
    segments = segment_eeg_signal(bipolar_data, window_size, step_size, resampling_frequency)
    indx = random.randint(0, len(segments)-1)
    data_5_min = segments[indx]
    #print('i am here in dataloader')
    #print(data_5_min.shape)


    if train:
        # Extract labels.
        outcome = int(get_outcome(patient_metadata))
        #print(outcome)
        cpc = int(get_cpc(patient_metadata))
        #print(cpc)

        x = data_5_min 

        outcome = torch.tensor(outcome, dtype=torch.long)
        outcome = outcome.to(device)

        cpc = torch.tensor(cpc, dtype=torch.long)
        cpc = cpc.to(device)
        #outcome = number_to_one_hot(outcome, 2)
        return x, outcome, cpc
    else:
        x = data_5_min
        return x
    
#%%
class dataset(Dataset):
    def __init__(self, config, data_folder, X_files, device, train=True):
        self.X_files = X_files
        self.train=train
        self.data_folder = data_folder
        self.device = device
        self.config = config 

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        patient_id = self.X_files[idx]
        x, outcome, cpc = load_data(self.config, self.data_folder, patient_id, self.device)
        return {"input":x, "outcome":outcome, "cpc": cpc/5.0}

#%%
class targets(Dataset):
    def __init__(self, data_folder, X_files, train=True):
        self.X_files = X_files
        self.train=train
        self.data_folder = data_folder

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        patient_id = self.X_files[idx]

        outcome = get_labels(self.data_folder, patient_id)
        
        return {"outcome":outcome}
