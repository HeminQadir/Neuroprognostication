#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#%%
# Libraries for the first entery
import numpy as np
import torch
from config_file import get_config 
from utils import * 

#%%

def train_challenge_model(data_folder, model_folder, verbose=2):
    # Required parameters
    
    config =  get_config()
    local_rank = -1
    seed = 42

    # Setup CUDA, GPU & distributed training
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        n_gpu = 1

    # Set seed
    set_seed(seed=seed, n_gpu=n_gpu)

    # Model & Tokenizer Setup
    model = setup(config, device)

    # Training
    train(config, model, data_folder, model_folder, device, local_rank, n_gpu)


#%%
# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = load_data(data_folder, patient_id, device, train=False)

    if len(x)>0:
        # Apply models to features.
        models.eval()
        outputs, pred_cpcs, _ = models(x.unsqueeze(0))
        
        outcome_probabilities = F.softmax(outputs[0])
                
        pred_outcome = torch.argmax(outcome_probabilities)
        outcome_probability = outcome_probabilities[1]   # predicted probability of a poor outcome
        outcome_probability = outcome_probability.data.cpu().item()
        pred_outcome = pred_outcome.data.cpu().item()

        pred_cpcs = pred_cpcs*5
        pred_cpcs = pred_cpcs.data.cpu().item()
        pred_cpcs = np.clip(pred_cpcs, 1, 5)  

        print("="*80)
        print(pred_outcome)
        print(pred_cpcs)
        #outcome_probability = round(outcome_probability, 2)
        print(outcome_probability)
        
    else:
        pred_outcome, outcome_probability, pred_cpcs = float(0), float(0), float(0) #float('nan'), float('nan'), float('nan')

    return pred_outcome, outcome_probability, pred_cpcs