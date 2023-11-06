#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#%%
# Libraries for the first entery
from helper_code import *
import numpy as np, os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import random
from models import VisionTransformer
import math
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from dataset_loader import *

#%%
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

#%%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self): 
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#%%
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


#%%
def set_seed(seed=42, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


#%%
def load_train_val_files(data_folder, split=True, split_ratio=0.1):

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    if split:
        X_train, X_val = train_test_split(patient_ids, test_size=split_ratio, 
                                        shuffle=True, random_state=42)
        return X_train, X_val
    else:
        X_train = patient_ids
        return X_train


#%%
def get_class_weights(targets):
    class_counts = torch.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (2 * class_counts)
    return class_weights


#%%
def get_upsampled_loader(targets):
    class_weights = get_class_weights(targets)
    samples_weights = class_weights[targets]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    return sampler

#%%
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

#%%
def setup(config, device):
    # Prepare model
    model = VisionTransformer(config, zero_head=True)
    model.to(device)
    num_params = count_parameters(model)
    #print(model)
    #print(num_params)
    return model

# Save your trained model.
def save_challenge_model(model_folder, outcome_model, epoch): #, imputer, outcome_model, cpc_model):
    torch.save({'model': outcome_model, 'epoch': epoch,}, os.path.join(model_folder, 'model.pt'))

def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.pt')
    state = torch.load(filename)
    model = state['model']
    return model 

#%%
def valid(model, val_loader, local_rank, device):
    # Validation!

    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for step, batch in enumerate(epoch_iterator):

        data = batch
        x, y, cpcs = data["input"].to(device), data["outcome"].to(device), data["cpc"].to(device)
    
        with torch.no_grad():
            logits, regression, _ = model(x) #[0]   #[0] not needed if we have only one output

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            #print(preds)

            predict = y == preds

            TP += (predict == True).sum()
            FP += (predict == False).sum()
            
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    #accuracy = simple_accuracy(all_preds, all_label)
    precision = TP / float(TP+FP) * 100

    print("precision: ", precision)

    return precision #accuracy


#%%
def train(config, model, data_folder, model_folder, device, local_rank, n_gpu):

    num_steps = config.num_steps
    eval_every = config.eval_every
    eval_batch_size = config.eval_batch_size
    learning_rate = config.learning_rate
    train_batch_size = config.train_batch_size 
    
    """ Train the model """
    name = "physionet"
    gradient_accumulation_steps = 1
    decay_type = "cosine" #choices=["cosine", "linear"]
    warmup_steps = 500 
    max_grad_norm = 1.0

    if local_rank in [-1, 0]:
        os.makedirs(model_folder, exist_ok=True)

    train_batch_size = train_batch_size // gradient_accumulation_steps

    split = True
    split_ratio = 0.1
    shuffle = True

    if split:
        X_train, X_val = load_train_val_files(data_folder, split, split_ratio)
        
        trainset = dataset(config, data_folder, X_train, device)
        label = targets(data_folder, X_train)
        train_labels = list()
        for i, data in enumerate(label):
            label = data["outcome"]
            train_labels.append(label.item())
        train_labels = torch.tensor(train_labels)
        sampler = get_upsampled_loader(train_labels)
        train_loader =  DataLoader(trainset, batch_size=train_batch_size, sampler=sampler) #shuffle=shuffle)

        valset = dataset(config, data_folder, X_val, device)
        val_loader =  DataLoader(valset, batch_size=eval_batch_size, shuffle=shuffle)

    else:
        X_train = load_train_val_files(data_folder, split, split_ratio)
        trainset = dataset(config, data_folder, X_train, device)
        label = targets(data_folder, X_train)
        train_labels = list()
        for i, data in enumerate(label):
            label = data["outcome"]
            train_labels.append(label.item())
        train_labels = torch.tensor(train_labels)
        sampler = get_upsampled_loader(train_labels)
        train_loader =  DataLoader(trainset, batch_size=train_batch_size, sampler=sampler) #shuffle=shuffle)

        #data_folder_val = "/media/jacobo/NewDrive/physionet.org/files/i-care/2.0/validation"
        #X_val = load_train_val_files(data_folder_val, split, split_ratio)
        #valset = dataset(config, data_folder_val, X_val)
        #val_loader =  DataLoader(valset, batch_size=eval_batch_size, shuffle=shuffle)
    

    weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                betas=(0.9, 0.999), 
                                eps=1e-08,
                                weight_decay=weight_decay)

    t_total = num_steps
    if decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)


    # Train!
    model.zero_grad()
    seed = 42
    set_seed(seed=seed, n_gpu=n_gpu)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            data = batch
            x, y, cpcs = data["input"].to(device), data["outcome"].to(device), data["cpc"].to(device)
            #print("I am label: ", y)
            loss1, loss2 = model(x, y, cpcs)

            loss = loss1 + loss2 

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                losses.update(loss.item()*gradient_accumulation_steps)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f) (loss_class=%2.5f) (loss_regress=%2.5f)" % (global_step, t_total, losses.val, loss1, loss2)
                )

                if global_step % eval_every == 0 and local_rank in [-1, 0]:
                    accuracy = valid(model, val_loader, local_rank, device)
                    if best_acc < accuracy:
                        #save_challenge_model(args, model)
                        save_challenge_model(model_folder, model, global_step)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break