# Project2 for EN.520.666 Information Extraction

# 2021 Ruizhe Huang
# 2022 Zili Huang

import numpy as np
import matplotlib.pyplot as plt
import string
import sys
import os
from HMM import HMM
from collections import Counter, defaultdict
from itertools import chain
from tqdm import tqdm
from scipy.special import logsumexp
import pickle


NOISE = "<noise>"
data_dir = ""
log_likelihoods = []

def read_file_line_by_line(file_name, func=lambda x: x, skip_header=True):
    print("reading file: %s" % file_name)
    res = list()
    with open(file_name, "r") as fin:
        if skip_header:
            fin.readline()  # skip the header
        for line in fin:
            if len(line.strip()) == 0:
                continue
            fields = func(line.strip())
            res.append(fields)
    print("%d lines, done" % len(res))
    return res


class Word_Recognizer:

    def __init__(self, restore_ith_epoch=None, split=False):
        # read labels
        self.lblnames = read_file_line_by_line(os.path.join(data_dir, "clsp.lblnames"))

        # read training data
        self.trnlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.trnlbls"), func=lambda x: x.split())
        self.endpts = read_file_line_by_line(os.path.join(data_dir, "clsp.endpts"), func=lambda x: list(map(int, x.split())))
        self.trnscr = read_file_line_by_line(os.path.join(data_dir, "clsp.trnscr"))

        # read dev data
        self.devlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.devlbls"), func=lambda x: x.split())
        self.train_words = set(self.trnscr)

        assert len(self.trnlbls) == len(self.endpts)
        assert len(self.trnlbls) == len(self.trnscr)

        # 23 letters + noise
        self.letters = list(string.ascii_lowercase)
        for c in ['k', 'q', 'z']:
            self.letters.remove(c)
        self.noise_id = len(self.letters)
        self.letters.append(NOISE)
        self.letter2id = dict({c: i for i, c in enumerate(self.letters)})
        self.id2letter = dict({i: c for c, i in self.letter2id.items()})

        # 256 quantized feature-vector labels
        self.label2id = dict({lbl: i for i, lbl in enumerate(self.lblnames)})
        self.id2label = dict({i: lbl for lbl, i in self.label2id.items()})

        # convert file contents to integer ids
        self.trnlbls = [[self.label2id[lbl] for lbl in line] for line in self.trnlbls]
        self.devlbls = [[self.label2id[lbl] for lbl in line] for line in self.devlbls]
        self.trnscr = [[self.letter2id[c] for c in word] for word in self.trnscr]

        # get label frequencies
        lbl_freq = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=0)
        lbl_freq_noise = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=1, endpts=self.endpts)

        # get hmms for each letter
        self.letter_id2hmm = self.init_letter_hmm(lbl_freq, lbl_freq_noise, self.id2letter)
        self.split = split

    def get_unigram(self, trnlbls, nlabels, smooth=0, endpts=None):
        # Compute "unigram" frequency of the training labels
        # Return freq(np array): the "unigram" frequency of the training labels
        
        freq = np.zeros(nlabels)
        for index, line in enumerate(trnlbls):
            if endpts == None:
                for label in line:
                    freq[label] += 1
            else:
                for label in line[:endpts[index][0]+1]:
                    freq[label] += 1
                for label in line[endpts[index][1]:]:
                    freq[label] += 1
        freq = freq + smooth
        freq = freq / np.sum(freq)
        return freq

    def init_letter_hmm(self, lbl_freq, lbl_freq_noise, id2letter):
        # Initialize the HMM for each letter
        # Return letter_id2hmm(dict): the key is the letter_id and the value
        # is the corresponding HMM
        transition_probs = np.asarray([[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 0.8]], dtype=np.float64)
        transition_probs_noise = np.asarray([[0.25, 0.25, 0.25, 0.25, 0.0], 
                                          [0.0, 0.25, 0.25, 0.25, 0.25], 
                                          [0.0, 0.25, 0.25, 0.25, 0.25], 
                                          [0.0, 0.25, 0.25, 0.25, 0.25], 
                                          [0.0, 0.0, 0.0, 0.0, 0.75]], dtype=np.float64)
        emission_probs = np.zeros((256, 3, 3))
        emission_probs_noise = np.zeros((256, 5, 5))

        for i in range(256):
            mask = (transition_probs > 0).astype(np.int32)
            noise_mask = (transition_probs_noise > 0).astype(np.int32)
            emission_probs[i, :, :] = lbl_freq[i]
            emission_probs_noise[i, :, :] = lbl_freq_noise[i]
            emission_probs[i, :, :] = emission_probs[i, :, :] * mask
            emission_probs_noise[i, :, :] = emission_probs_noise[i, :, :] * noise_mask
        
        letter_id2hmm = {}
        for letter_id in range(len(self.letters)):
            if letter_id != self.noise_id:
                letter_id2hmm[letter_id] = HMM(num_states=3, num_outputs=256)
                letter_id2hmm[letter_id].init_transition_probs(transition_probs)
                letter_id2hmm[letter_id].init_emission_probs(emission_probs)
            else:
                letter_id2hmm[letter_id] = HMM(num_states=5, num_outputs=256)
                letter_id2hmm[letter_id].init_transition_probs(transition_probs_noise)
                letter_id2hmm[letter_id].init_emission_probs(emission_probs_noise)
        return letter_id2hmm

    def id2word(self, w):
        # w should be a list of char ids
        return ''.join(map((lambda c: self.id2letter[c]), w))

    def get_word_model(self, scr):
        # Construct the word HMM based on self.letter_id2hmm
        # Return h(HMM object): the word HMM for the word scr 
        # Construct the word HMM by concatentating the letter HMMs
        
        h = HMM(num_states=10+3*len(scr), num_outputs=256)
        h_transition_probs = np.zeros((10+3*len(scr), 10+3*len(scr)))
        h_emission_probs = np.zeros((256, 10+3*len(scr), 10+3*len(scr)))

        h_transition_probs[:5, :5] = self.letter_id2hmm[self.noise_id].transitions
        h_emission_probs[:, :5, :5] = self.letter_id2hmm[self.noise_id].emissions
        h_transition_probs[-5:, -5:] = self.letter_id2hmm[self.noise_id].transitions
        h_emission_probs[:, -5:, -5:] = self.letter_id2hmm[self.noise_id].emissions
        
        null_arc_dict = defaultdict(dict)
        null_arc_dict[4][5] = 0.25
        for index, letter in enumerate(scr):
            h_transition_probs[5+index*3:8+index*3, 5+index*3:8+index*3] = self.letter_id2hmm[letter].transitions
            h_emission_probs[:, 5+index*3:8+index*3, 5+index*3:8+index*3] = self.letter_id2hmm[letter].emissions
            null_arc_dict[4+(index+1)*3][5+(index+1)*3] = 0.2
        
        h.init_transition_probs(h_transition_probs)
        h.init_emission_probs(h_emission_probs)
        h.init_null_arcs(null_arc_dict)
        return h


    def update_letter_counters(self, scr, word_hmm):
        # Update self.letter_id2hmm based on the counts from word_hmm

        self.letter_id2hmm[self.noise_id].set_counters(word_hmm.output_arc_counts[:, :5, :5], word_hmm.output_arc_counts_null)
        self.letter_id2hmm[self.noise_id].set_counters(word_hmm.output_arc_counts[:, -5:, -5:], 
                                                       word_hmm.output_arc_counts_null)
        
        for index, letter_id in enumerate(scr):
            self.letter_id2hmm[letter_id].set_counters(
                word_hmm.output_arc_counts[:, 5+index*3:8+index*3, 5+index*3:8+index*3],
                word_hmm.output_arc_counts_null)

    def train(self, num_epochs=1):

        # sort trnlbls, endpts and trnscr such that the same word appear next to each other
        trnlbls_sorted = []
        trnscr_sorted = []
        heldoutlbls_sorted = []
        heldoutscr_sorted = []


        if self.split:
            heldoutscr = self.trnscr[:int(len(self.trnscr)*0.2)]
            heldoutlbls = self.trnlbls[:int(len(self.trnlbls)*0.2)]
            self.trainscr = self.trnscr[int(len(self.trnscr)*0.2):]
            self.trainlbls = self.trnlbls[int(len(self.trnlbls)*0.2):]
            for scr, lbls in sorted(zip(heldoutscr, heldoutlbls)):
                heldoutlbls_sorted.append(lbls)
                heldoutscr_sorted.append(scr)
            for scr, lbls in sorted(zip(self.trnscr, self.trnlbls)):
                trnlbls_sorted.append(lbls)
                trnscr_sorted.append(scr)
        
        else:
            for scr, lbls in sorted(zip(self.trnscr, self.trnlbls)):
                trnlbls_sorted.append(lbls)
                trnscr_sorted.append(scr)
        
        # Training
        for i_epoch in range(num_epochs):
            log_likelihood = 0
            num_frames = 0

            for letter_id in self.letter_id2hmm:
                self.letter_id2hmm[letter_id].reset_counters()

            print("---- echo: %d ----" % i_epoch)
            for scr, lbls in zip(trnscr_sorted, trnlbls_sorted):
                word_hmm = self.get_word_model(scr)
                init_prob = np.asarray([1]+[0]*(word_hmm.num_states-1), dtype=np.float64)

                alpha, beta, q = word_hmm.forward_backward(lbls, init_prob=init_prob, update_params=False)
                self.update_letter_counters(scr, word_hmm)
               
                log_likelihood += word_hmm.compute_log_likelihood(lbls, init_prob=init_prob, init_beta=np.asarray([1]*word_hmm.num_states))
                num_frames += len(lbls)

            #update parameters of each letter hmm
            for letter_id in self.letter_id2hmm:
               self.letter_id2hmm[letter_id].update_params()
            
            for letter_id in range(len(self.letters)):
                self.letter_id2hmm[letter_id].reset_counters()
            
            print("log_likelihood =", log_likelihood, "per_frame_log_likelihood =", log_likelihood / num_frames)
            log_likelihoods.append(log_likelihood)
            #self.save(i_epoch)
            if self.split:
                self.test_heldout(heldoutlbls_sorted, heldoutscr_sorted)
        self.test()

    def test(self):
        # Compute the word likelihood for each dev samples 
        id2words = dict({i: w for i, w in enumerate(self.train_words)})
        words2id = dict({w: i for i, w in id2words.items()})
        word_likelihoods = np.zeros((len(words2id), len(self.devlbls)))
        for i, lbls in enumerate(self.devlbls):
            for j, word in enumerate(self.train_words):
                scr = [self.letter2id[letter] for letter in word]
                word_hmm = self.get_word_model(scr)
                init_prob = np.zeros((word_hmm.num_states), dtype=np.float64)
                init_prob[0] = 1.0
                word_likelihoods[j, i] = word_hmm.compute_log_likelihood(lbls, init_prob=init_prob, init_beta=np.ones((word_hmm.num_states)))
        
        result = word_likelihoods.argmax(axis=0)
        prediction = [id2words[res] for res in result]
        confidence = np.exp(word_likelihoods.max(axis=0) - logsumexp(word_likelihoods, axis=0))
        confidence[np.isnan(confidence)] = 1.0

        file = open("result.txt", "a+")
        print(f"Prediction: {prediction}", file=file)
        print(f"Confidence: {confidence}", file=file)

    def test_heldout(self, heldoutlbls, heldoutscr):
        id2words = dict({i: w for i, w in enumerate(self.train_words)})
        words2id = dict({w: i for i, w in id2words.items()})
        word_likelihoods = np.zeros((len(words2id), len(heldoutlbls)))
        for i, lbls in enumerate(heldoutlbls):
            for j, word in enumerate(self.train_words):
                scr = [self.letter2id[letter] for letter in word]
                word_hmm = self.get_word_model(scr)
                init_prob = np.zeros((word_hmm.num_states), dtype=np.float64)
                init_prob[0] = 1.0
                word_likelihoods[j, i] = word_hmm.compute_log_likelihood(lbls, init_prob=init_prob, init_beta=np.ones((word_hmm.num_states)))
        result = word_likelihoods.argmax(axis=0)
        prediction = [id2words[res] for res in result]
        groundtruth = [''.join([self.id2letter[letter_id] for letter_id in scr]) for scr in heldoutscr]
        print(prediction[:10])
        print(groundtruth[:10])
        accuracy = sum([1 if prediction[i] == groundtruth[i] else 0 for i in range(len(prediction))]) / len(prediction)
        print("Accuracy: ", accuracy)

    def save(self, i_epoch):
        fn = os.path.join(data_dir, "%d.mdl.pkl" % i_epoch)
        print("Saved to:", fn)
        for letter_id, hmm in self.letter_id2hmm.items():
            hmm.output_arc_counts = None
            hmm.output_arc_counts_null = None
        pickle.dump(self.letter_id2hmm, open(fn, "wb"))

    def load(self, i_epoch):
        return pickle.load(open(os.path.join(data_dir, "%d.mdl.pkl" % i_epoch), "rb"))


def main(args):
    n_epochs = 10
    if len(args) > 1 and args[1] == "split":
        split = True
    else:
        split = False

    wr = Word_Recognizer(split=split)
    wr.train(num_epochs=n_epochs)

    plt.figure()
    plt.plot(range(n_epochs), log_likelihoods, marker='o')
    plt.title("Log likelihood per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Log likelihood")
    plt.savefig(f"log_likelihood.pdf")

if __name__ == '__main__':
    args = sys.argv    
    main(args)
