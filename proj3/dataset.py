import string
import torch
import librosa
import os
import pandas as pd
from torch.utils.data import Dataset

class AsrDataset(Dataset):
    def __init__(self, scr_file, feature_type='discrete', feature_file=None,
                 feature_label_file=None,
                 wav_scp=None, wav_dir=None):
        """
        :param scr_file: clsp.trnscr
        :param feature_type: "discrete" or "mfcc"
        :param feature_file: clsp.trainlbls or clsp.devlbls
        :param feature_label_file: clsp.lblnames
        :param wav_scp: clsp.trnwav or clsp.devwav
        :param wav_dir: wavforms/
        """
        self.feature_type = feature_type
        assert self.feature_type in ['discrete', 'mfcc']

        self.blank = "<blank>"
        self.silence = "<sil>"
        
        # === write your code here ===
    
        self.letters = list(string.ascii_lowercase) + [self.silence, self.blank]
        self.letter2idx = {letter: idx for idx, letter in enumerate(self.letters)}
        self.idx2letter = {idx: letter for idx, letter in enumerate(self.letters)}
        self.words = {}

        print(f"silence index = {self.letter2idx[self.silence]}")
        print(f"blank index = {self.letter2idx[self.blank]}")

        # read scr_file
        with open(scr_file, 'r') as f:
            # remove the first line
            f.readline()
            self.script = [line.strip() for line in f.readlines()]
        
        # read feature_file
        if feature_type == 'discrete':
            with open(feature_file, 'r') as f:
                # remove the first line
                f.readline()
                self.features = [line.strip().split() for line in f.readlines()]

        elif feature_type == 'mfcc':
            self.features = self.compute_mfcc(wav_scp, wav_dir)

        # read feature_label_file
        with open(feature_label_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
            #remove fist line
            self.labels.pop(0)
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx2label = {idx: label for idx, label in enumerate(self.labels)}
        print(f"labels = {self.labels}")
        print(f"num labels = {len(self.labels)}")

    def __len__(self):
        """
        :return: num_of_samples
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get one sample each time. Do not forget the leading- and trailing-silence.
        :param idx: index of sample
        :return: spelling_of_word, feature
        """
        # === write your code here === 
        spelling_of_word = self.script[idx]
        spelling_of_word = [self.letter2idx[self.silence]] + [self.letter2idx[c] for c in spelling_of_word] + [self.letter2idx[self.silence]]
        
        if self.script[idx] not in self.words:
            self.words[self.script[idx]] = spelling_of_word
    
        if self.feature_type == 'discrete':
            feature = [self.label2idx[c] for c in self.features[idx]] 
        elif self.feature_type == 'mfcc':
            feature = self.features[idx]
        return spelling_of_word, feature

    # This function is provided
    def compute_mfcc(self, wav_scp, wav_dir):
        """
        Compute MFCC acoustic features (dim=40) for each wav file.
        :param wav_scp:
        :param wav_dir:
        :return: features: List[np.ndarray, ...]
        """
        features = []
        with open(wav_scp, 'r') as f:
            for wavfile in f:
                wavfile = wavfile.strip()
                if wavfile == 'jhucsp.trnwav' or wavfile == 'jhucsp.devwav':  # skip header
                    continue
                wav, sr = librosa.load(os.path.join(wav_dir, wavfile), sr=None)
                feats = librosa.feature.mfcc(y=wav, sr=16e3, n_mfcc=40, hop_length=160, win_length=400).transpose()
                features.append(feats)
        return features
