# Project1 for EN.520.666 Information Extraction

# 2021 Matthew Ost
# 2021 Ruizhe Huang
# 2022 Zili Huang

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import string

EPS = 1e-6

def load_data(fname):
    
    alphabet_string = string.ascii_lowercase
    char_list = list(alphabet_string)
    print(char_list)
    freq_list = [0 for _ in range(27)]
    char_list.append(' ')
    with open(fname, 'r') as fh:
        content = fh.readline()
    content = content.strip('\n')
    data = []
    for c in content:
        assert c in char_list
        data.append(char_list.index(c))
        freq_list[char_list.index(c)] += 1

    #freq_list = [count / np.sum(freq_list) for count in freq_list]
    #plt.figure(1)
    #plt.bar(char_list, freq_list)
    #plt.xlabel('characters')
    #plt.ylabel('frequency')
    #plt.savefig('freq.pdf')
    
    return np.array(data)#, freq_list

def get_init_prob_2states():
    
    #r_y = np.random.randint(0, high=10,size=27)
    #delta_y = r_y - np.mean(r_y)
    #E_prob_1 = freq_list - 0.0001*delta_y
    #E_prob_2 = freq_list + 0.0001*delta_y
    
    #E_prob_new = np.array([E_prob_1, E_prob_2])
    # Define initial transition probability and emission probability
    # for 2 states HMM
    T_prob = np.array([[0.49, 0.51], [0.51, 0.49]])
    E_prob = np.zeros((2, 27))
    E_prob[0, 0:13] =  0.0370
    E_prob[1, 13:27] =  0.0370
    E_prob[0, 13:27] =  0.0371
    E_prob[1, 0:13] =  0.0371
    E_prob[1, 26] = E_prob[0, 26] = 0.0367
    return T_prob, E_prob

def get_init_prob_4states():
    
    T_prob = np.array([[0.24, 0.26, 0.24, 0.26], [0.26, 0.24, 0.26, 0.24], [0.26, 0.26, 0.24, 0.24], [0.24, 0.24, 0.26, 0.26]])
    E_prob = np.zeros((4, 27))
    E_prob[0, 0:13] =  0.0370
    E_prob[1, 13:27] =  0.0370
    E_prob[0, 13:27] =  0.0371
    E_prob[1, 0:13] =  0.0371
    E_prob[1, 26] = E_prob[0, 26] = 0.0367
    E_prob[2, 0:13] =  0.0370
    E_prob[3, 13:27] =  0.0370
    E_prob[2, 13:27] =  0.0371
    E_prob[3, 0:13] =  0.0371
    E_prob[3, 26] = E_prob[2, 26] = 0.0367
    return T_prob, E_prob

class HMM:

    def __init__(self, num_states, num_outputs):
        # Args:
        #     num_states (int): number of HMM states
        #     num_outputs (int): number of output symbols            

        self.states = np.arange(num_states)  # just use all zero-based index
        self.outputs = np.arange(num_outputs)
        self.num_states = num_states
        self.num_outputs = num_outputs

        # Probability matrices
        self.transitions = None
        self.emissions = None

    def initialize(self, T_prob, E_prob):
        # Initialize HMM with transition probability T_prob and emission probability
        # E_prob

        # Args:
        #     T_prob (numpy.ndarray): [num_states x num_states] numpy array.
        #     T_prob[i, j] is the transition probability from state i to state j.
        #     E_prob (numpy.ndarray): [num_states x num_outputs] numpy array.
        #     E_prob[i, j] is the emission probability of state i to output jth symbol. 
        self.transitions = T_prob
        self.emissions = E_prob
        self._assert_transition_probs()
        self._assert_emission_probs()

    def _assert_emission_probs(self):
        for s in self.states:
            assert self.emissions[s].sum() - 1 < EPS

    def _assert_transition_probs(self):
        for s in self.states:
            assert self.transitions[s].sum() - 1 < EPS
            assert self.transitions[:, s].sum() - 1 < EPS

    def forward(self, obs):
        # Compute the forward pass of the HMM
        m = self.num_states
        T = len(obs)
        alpha = np.zeros((m, T))
        
        init = np.array([1/m for _ in range(m)]) # uniform initial state distribution
        alpha[:, 0] = init * self.emissions[:, obs[0]]
        q = [1]
        # alpha[:, 0] = alpha[:, 0] / q[0]

        for t in range(1, T):
            for z_t in range(m):
                for z_t_minus_one in range(m):
                    alpha[z_t, t] += alpha[z_t_minus_one, t-1] * self.transitions[z_t_minus_one, z_t] * self.emissions[z_t, obs[t]]
            # Q-Normalization 
            q_t = np.sum(alpha[:, t])
            q.append(q_t)
            alpha[:, t] = alpha[:, t] / q_t
        return alpha, q

    def backward(self, obs, q):
        # Compute the backward pass of the HMM
        m = self.num_states
        T = len(obs)

        beta = np.zeros((m, T))
        beta[:, T-1] = 1

        for t in range(T-2, -1, -1):
            for z_t in range(m):
                for z_t_plus_one in range(m):
                    beta[z_t, t] += beta[z_t_plus_one, t+1] * self.transitions[z_t, z_t_plus_one] * self.emissions[z_t_plus_one, obs[t+1]]
            # Q-Normalization
            beta[:, t] = beta[:, t] / q[t+1]
        return beta    

    def Baum_Welch(self, max_iter, train_data, test_data):
        # The Baum Welch algorithm to estimate HMM parameters
        # Args:
        #     max_iter (int): maximum number of iterations to train
        #     train_data (numpy.ndarray): train data
        #     test_data (numpy.ndarray): test data
        #
        # Returns:
        #     info_dict (dict): dictionary containing information to visualize

        info_dict = {'iteration': [],
                    'train': [], 
                    'test': [], 
                    'emission_a': [[],[],[],[]],
                    'emission_n': [[],[],[],[]]}

        for it in range(max_iter):
            # Implement the Baum-Welch algorithm here

            # The forward pass
            alpha, q = self.forward(train_data)  
            # The backward pass
            beta = self.backward(train_data, q)

            # Transition probability evaluation
            log_prob = np.sum(np.log(q))
            
            info_dict['iteration'].append(it)
            info_dict['train'].append(self.log_likelihood(train_data))
            info_dict['test'].append(self.log_likelihood(test_data))
            
            for i in range(self.num_states):
                info_dict['emission_a'][i].append(self.emissions[i, 0])
                info_dict['emission_n'][i].append(self.emissions[i, 13])

            print(f"iteration {it} | train_likelihood {info_dict['train'][-1]} | test_likelihood {info_dict['test'][-1]}")
 
            # print("prob: ", log_prob)
            T = len(train_data)
            
            expected_counts = np.zeros((self.num_states, self.num_states))
            for t in range(T-1):
                for i in range(self.num_states):
                    for j in range(self.num_states):
                        #if i != j:
                        expected_counts[i, j] +=  alpha[i, t] * self.transitions[i, j] * self.emissions[j, train_data[t+1]] * beta[j, t+1] / q[t+1]
                        #else:
                        #    expected_counts[i, j] += alpha[i, t] * self.transitions[i, j] * self.emissions[j, train_data[t+1]] * beta[j, t+1] / q[t+1]
            
            # Parameter Estimation
            updated_transition = np.zeros((self.num_states, self.num_states))
            updated_emission = np.zeros((self.num_states, self.num_outputs))
            for i in range(self.num_states):
                for j in range(self.num_states):
                    updated_transition[i, j] = np.sum(expected_counts[i, j]) / np.sum(expected_counts[i, :])

            self.transitions = updated_transition
            #print(self.transitions)
            gamma = np.multiply(alpha, beta) 
            for i in range(self.num_states):
                for j in range(self.num_outputs):
                    updated_emission[i, j] = np.sum(gamma[i, train_data == j]) / np.sum(gamma[i, :])
            self.emissions = updated_emission
            #print(self.emissions)
            #print(f"q(n|1) {self.emissions[0, 13]} | q(n|2) {self.emissions[1, 13]} | q(a|1) {self.emissions[0, 0]} | q(a|2) {self.emissions[1, 0]}")
        #print(self.emissions[0, :])
        #print(self.emissions[1, :])
        
        alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '#']
        
        #for i in range(5, 9):
        #    plt.figure(i)
        #    plt.bar(alphabets, self.emissions[i-5, :])
        #    plt.xlabel('Alphabets')
        #    plt.ylabel('Emission Probability')
        #    plt.title(f'Emission Probability bar plot of state {i-4}')
        #    plt.savefig(f"state{i-4}_4.pdf")

        return info_dict

    def log_likelihood(self, data):
        # Compute the log likelihood of sequence data
        # Args:
        #     data (numpy.ndarray): 
        #
        # Returns:
        #     prob (float): log likelihood of data
        
        T = len(data)
        m = self.num_states
        alpha, q = self.forward(data)
        prob = np.sum(np.log(q))
        return prob/len(data)
        #return np.log(prob)/len(data)

    def visualize(self, info_dict):
        # Visualize
        plt.figure(1)
        plt.plot(info_dict['iteration'], info_dict['train'], label='train')
        plt.xlabel('iteration')
        plt.ylabel('train log likelihood')
        
        plt.figure(1)
        plt.plot(info_dict['iteration'], info_dict['test'], label='test')
        plt.xlabel('iteration')
        plt.ylabel('test log likelihood')
        plt.savefig('train_test_4_states.pdf')

        plt.figure(3)
        plt.plot(info_dict['iteration'], info_dict['emission_a'][0], label='emission_a_1', color='r')
        plt.plot(info_dict['iteration'], info_dict['emission_a'][1], label='emission_a_2', color='b')
        #plt.plot(info_dict['iteration'], info_dict['emission_a'][2], label='emission_a_3', color='c')
        #plt.plot(info_dict['iteration'], info_dict['emission_a'][3], label='emission_a_4', color='y')
        plt.xlabel('iteration')
        plt.ylabel('emission probability of a')
        plt.legend(loc='upper right')
        plt.savefig('emission_4_states_a.pdf')
        
        plt.figure(4)
        plt.plot(info_dict['iteration'], info_dict['emission_n'][0], label='emission_a_1', color='r')
        plt.plot(info_dict['iteration'], info_dict['emission_n'][1], label='emission_a_2', color='b')
        #plt.plot(info_dict['iteration'], info_dict['emission_n'][2], label='emission_a_3', color='c')
        #plt.plot(info_dict['iteration'], info_dict['emission_n'][3], label='emission_a_4', color='y')
        plt.xlabel('iteration')
        plt.ylabel('emission probability of n')
        plt.legend(loc='upper right')
        plt.savefig('emission_4_states_n.pdf')
def main():
    n_states = 2
    n_outputs = 27
    train_file, test_file = "textA.txt", "textB.txt"
    max_iter = 600

    ## define initial transition probability and emission probability
    T_prob, E_prob = get_init_prob_2states() 
    #T_prob, E_prob = get_init_prob_4states()
    
    ## initial the HMM class
    H = HMM(num_states=n_states, num_outputs=n_outputs)

    ## initialize HMM with the transition probability and emission probability
    H.initialize(T_prob, E_prob)

    # load text file
    train_data, test_data = load_data(train_file), load_data(test_file)
    
    #train_data, _ = load_data(train_file)
    #test_data, _ = load_data(test_file)
    ## train the parameters of HMM
    info_dict = H.Baum_Welch(max_iter, train_data, test_data)

    ## visualize
    H.visualize(info_dict)

if __name__ == "__main__":
    main()
