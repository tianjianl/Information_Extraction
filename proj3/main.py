#!/usr/bin/env python

# 2022 Dongji Gao
# 2022 Yiwen Shao

import os
import sys
import string
import torch
import time
import re
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataset import AsrDataset
from model import LSTM_ASR


def collate_fn(batch):
    """
    This function will be passed to your dataloader.
    It pads word_spelling (and features) in the same batch to have equal length.with 0.
    :param batch: batch of input samples
    :return: (recommended) padded_word_spellings, 
                           padded_features,
                           list_of_unpadded_word_spelling_length (for CTCLoss),
                           list_of_unpadded_feature_length (for CTCLoss)
    """
    # === write your code here ===
    padded_word_spellings = pad_sequence([torch.tensor(sample[0]) for sample in batch], batch_first=True, padding_value=0)
    padded_features = pad_sequence([torch.tensor(sample[1]) for sample in batch], batch_first=True, padding_value=0)
    list_of_unpadded_word_spelling_length = [len(sample[0]) for sample in batch]
    list_of_unpadded_feature_length = [len(sample[1]) for sample in batch]

    return {
        "padded_word_spellings": padded_word_spellings,
        "padded_features": padded_features,
        "list_of_unpadded_word_spelling_length": list_of_unpadded_word_spelling_length,
        "list_of_unpadded_feature_length": list_of_unpadded_feature_length,
    }

def train(train_dataloader, model, ctc_loss, optimizer):
    # === write your code here ===
    model.train()
    start_time = time.time()

    for idx, data in enumerate(train_dataloader):
        padded_word_spellings = data["padded_word_spellings"]
        padded_features = data["padded_features"]
        
        #print(padded_features[0])
        list_of_unpadded_word_spelling_length = data["list_of_unpadded_word_spelling_length"]
        list_of_unpadded_feature_length = data["list_of_unpadded_feature_length"]
        output = model(padded_features)
        output = torch.transpose(output, 0, 1)
      
        loss = ctc_loss(output, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 5 == 0:            
            print(f"idx: {idx} loss: {loss.item()}")

    end_time = time.time()
    print(f"Training time: {end_time - start_time}s")


def decode(test_dataloader, model, idx2letter, blank_idx, prev_best, method='greedy', words=None):
    # === write your code here ===
    print(f"total number of test samples: {len(test_dataloader.dataset)}")
    print(f"total number of words: {len(words)}")
    model.eval()
    preds = []
    acts = []
    for idx, batch in enumerate(test_dataloader):
        padded_features = batch["padded_features"]
        output = model(padded_features)
        output = torch.transpose(output, 0, 1)
        if method == 'greedy':
            _, indices = torch.max(output, dim=2)
            for i in range(len(indices)):
                pred = ''.join([idx2letter[idx.item()] for idx in indices[i]])
                pred = pred.replace('<sil>', '')
                #collapse repeated characters between <blank>
                raw = pred.split('<blank>')
                pred = ''
                for r in raw:
                    #remove repeated characters
                    r = re.sub(r'(.)\1+', r'\1', r)
                    pred += r
                preds.append(pred)
                acts.append("".join([idx2letter[idx.item()] for idx in batch["padded_word_spellings"][i]]).replace('<sil>', ''))

        elif method == 'ctc':
        # compute ctc loss for every padded word spelling 
        # and find the one with the lowest loss   
            ctc_loss = nn.CTCLoss(blank=blank_idx)
            lowest_loss = float('inf')
            best_pred = ''
            for word in words:
                padded_word_spellings = torch.tensor([words[word]])
                list_of_unpadded_word_spelling_length = [len(words[word])]
                list_of_unpadded_feature_length = batch["list_of_unpadded_feature_length"]

                loss = ctc_loss(output, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
                if loss < lowest_loss:
                    lowest_loss = loss
                    best_pred = word
            preds.append(best_pred)
            acts.append("".join([idx2letter[idx.item()] for idx in batch["padded_word_spellings"][0]]).replace('<sil>', '').replace(' ', ''))

    print(f"prediction {preds[0:10]}")
    print(f"groundtruth {acts[0:10]}")
    acc = sum([1 if pred == act else 0 for pred, act in zip(preds, acts)]) / len(preds)
    print(f"accuracy: {acc}, prev_best {prev_best}")
    if acc > prev_best:
        prev_best = acc
        torch.save(model.state_dict(), f"model_{method}_best.pth")
        print(f"saved model with accuracy {acc}")
    return prev_best, preds

def compute_accuracy(test_dataloader, preds, idx2letter):
    # === write your code here ===
    print(f"len(preds): {len(preds)}")
    acts = []

    for idx, data in enumerate(test_dataloader):
        padded_word_spellings = data["padded_word_spellings"]
        list_of_unpadded_word_spelling_length = data["list_of_unpadded_word_spelling_length"]
        for word, length in zip(padded_word_spellings, list_of_unpadded_word_spelling_length):
            word_with_sil = ''.join([idx2letter[idx.item()] for idx in word[:length]])
            word_without_sil = word_with_sil.replace('<sil>', '')
            acts.append(word_without_sil)        
            #print(word_without_sil)
    print(f"len(acts): {len(acts)}")

def main(args):

    training_set = AsrDataset(scr_file='clsp.trnscr', feature_type=args[4],
                              feature_file='clsp.trnlbls', feature_label_file='clsp.lblnames', wav_scp='clsp.trnwav', wav_dir='waveforms')
    test_set = AsrDataset(scr_file='clsp.trnscr', feature_type=args[4],
                          feature_file='clsp.devlbls', feature_label_file='clsp.lblnames', wav_scp='clsp.devwav', wav_dir='waveforms')   

    idx2letter = training_set.idx2letter

    #split training set into training and validation set
    train_set, val_set = torch.utils.data.random_split(training_set, [748, 50])


    train_dataloader = DataLoader(train_set, batch_size=int(args[1]), shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)    
    test_dataloader =  DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = LSTM_ASR(feature_type=args[4], output_size=len(training_set.letters))

    # your can simply import ctc_loss from torch.nn
    loss_function = nn.CTCLoss(blank=training_set.blank_idx)

    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Training
    num_epochs = int(args[2])  
    prev_best = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train(train_dataloader, model, loss_function, optimizer)
        prev_best, _ = decode(val_dataloader, model, idx2letter, training_set.blank_idx, prev_best, method=args[3], words=training_set.words)
        #compute_accuracy(test_dataloader, preds, idx2letter)

    # Testing (totally by yourself)
    # preds = decode(test_dataloader, model, idx2letter)
    model.load_state_dict(torch.load(f"model_{args[3]}_best.pth"))
    _, preds = decode(test_dataloader, model, idx2letter, training_set.blank_idx, prev_best, method=args[3], words=training_set.words)
    
    with open('predictions_{args[3]}_{args[4]}.txt', 'a+') as f:
        for pred in preds:
            f.write(pred + '\n')
    
    # Evaluate (totally by yourself)
    # compute_accuracy(test_dataloader, preds, idx2letter)


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 4:
        print("Usage: python main.py <batch_size> <num_epochs> <method>")
        print("Supported method includes: greedy, ctc")
        exit(0)
    print(f"batch size = {args[1]}")
    print(f"num of epochs = {args[2]}")
    print(f"decode method = {args[3]}")
    if len(args) > 4:
        if args[4] == 'mfcc':
            feature_type = 'mfcc'
            print(f"feature_type = mfcc")
        else:
            feature_type = 'discrete'
            print(f"feature_type = discrete")
    else:
        feature_type = 'discrete'
        print(f"feature_type = discrete")
    args.append(feature_type)
    main(args)
