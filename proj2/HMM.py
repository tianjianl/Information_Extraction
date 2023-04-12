# Project1 for EN.520.666 Information Extraction

# 2021 Ruizhe Huang

import numpy as np
from scipy.special import logsumexp
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")


class HMM:

    def __init__(self, num_states, num_outputs):
        # Potentially useful constants for state and output alphabets
        self.states = range(num_states)  # just use all zero-based index
        self.outputs = range(num_outputs)
        self.num_states = num_states
        self.num_outputs = num_outputs

        # Probability matrices
        self.transitions = None  # key (1, 2) --> prob of s1 going to s2, i.e. p(s1 --> s2) = p(s2|s1) = p(t1)
        self.emissions = None  # key (3, 1, 2) --> prob of emitting 3 from arc (1->2), i.e. p(3|1->2)

        self.non_null_arcs = []  # a list of (ix, iy), where ix->iy is a non-null arc
        # TODO: can it be implemented as an matrix?
        self.null_arcs = dict()  # a dict of d[ix][iy]=p, where ix->iy is a null arc with probability p
        self.topo_order = []  # a list of states in the topo order that we can evaluate alphas properly

        self.output_arc_counts = None
        self.output_arc_counts_null = None

    def init_transition_probs(self, transitions):
        """Initialize transition probability matrices"""
        assert self.transitions is None
        self.transitions = transitions
        self._assert_transition_probs()

        for ix, iy in np.ndindex(self.transitions.shape):
            if self.transitions[ix, iy] - 0 > 10e-6:
                if iy < self.num_states:
                    self.non_null_arcs.append((ix, iy))

    def init_emission_probs(self, emission):
        """Initialize emission probability matrices"""
        assert self.emissions is None
        self.emissions = emission
        self._assert_emission_probs()

    def init_null_arcs(self, null_arcs=None):
        if null_arcs is not None:
            self.null_arcs = null_arcs  # note: null_arcs should be a dict

        # topo sort
        count = np.zeros(self.num_states)
        for ix in self.null_arcs.keys():
            for iy in self.null_arcs[ix].keys():
                if iy < self.num_states:
                    count[iy] += 1
        stack = [s for s in self.states if count[s] == 0]
        while len(stack) > 0:
            s = stack.pop()
            self.topo_order.append(s)
            if s not in self.null_arcs:
                continue
            for s_y in self.null_arcs[s]:
                count[s_y] -= 1
                if count[s_y] == 0:
                    stack.append(s_y)
        assert len(self.topo_order) == self.num_states

    def add_null_arc(self, ix, iy, prob):
        if self.null_arcs is None:
            self.null_arcs = dict()

        if ix not in self.null_arcs:
            self.null_arcs[ix] = dict()

        self.null_arcs[ix][iy] = prob

    def _assert_emission_probs(self):
        emission_sum = self.emissions.sum(axis=0)
        for arc in self.non_null_arcs:
            assert emission_sum[arc].sum() - 1 < 10e-6

    def _assert_transition_probs(self):
        for s in self.states:  # except the last state
            null_sum = 0 if s not in self.null_arcs else sum(self.null_arcs[s].values())
            assert self.transitions[s].sum() + null_sum - 1 < 10e-6

    def forward(self, data, init_prob=None):
        # Construct trellis for the forward pass with equally likely initial (stage-0) values, or given init_prob values

        alphas_ = np.zeros((len(data) + 1, self.num_states))  # normalized alphas
        Q = np.ones(len(data) + 1)  # Normalization constants to prevent underflow

        if init_prob is None:  # then assume uniform distribution
            init_prob = np.asarray([1 / self.num_states] * self.num_states)

        # Assumption: there is no null arc at the first stage or last stage
        alphas_[0] = init_prob
        Q[0] = alphas_[0].sum()
        alphas_[0] /= Q[0]

        # Begin forward pass
        for t in range(1, len(data) + 1):
            # Calculate alpha values for each state in this stage
            obs = data[t - 1]   # Note: alpha[t] actually means the prob of generating data[0: t-1]

            # non_null arcs
            alphas_[t] = np.dot(alphas_[t - 1], self.transitions * self.emissions[obs])
            # null arcs, except the final stage
            if t < len(data):
                for s in self.topo_order:
                    if s not in self.null_arcs:
                        continue
                    for s_y in self.null_arcs[s]:
                        alphas_[t][s_y] += alphas_[t][s] * self.null_arcs[s][s_y]

            # Compute next Q factor and normalize alphas in this stage by Qi
            Q[t] = alphas_[t].sum()
            alphas_[t] /= Q[t]

        # print("alphas_", alphas_)
        # print("Q", Q)
        return alphas_, Q

    def forward_log(self, data, init_log_prob=None):
        # Construct trellis for the forward pass with equally likely initial (stage-0) values, or given init_prob values

        log_alphas = np.empty((len(data) + 1, self.num_states))

        if init_log_prob is None:  # then assume uniform distribution
            init_log_prob = np.asarray([1 / self.num_states] * self.num_states)
            init_log_prob = np.log(init_log_prob)

        log_alphas[0] = init_log_prob

        # Begin forward pass
        for t in range(1, len(data) + 1):
            # Calculate alpha values for each state in this stage
            obs = data[t - 1]   # Note: alpha[t] actually means the prob of generating data[0: t-1]
            trans_matrix = self.transitions * self.emissions[obs]
            # non_null arcs
            for j in range(self.num_states):
                log_alphas[t][j] = logsumexp(log_alphas[t - 1] + np.log(trans_matrix[:, j]))
            # null arcs, except the final stage
            if t < len(data):
                for s in self.topo_order:
                    if s not in self.null_arcs:
                        continue
                    for s_y in self.null_arcs[s]:
                        log_alphas[t][s_y] = logsumexp([
                            log_alphas[t][s_y],
                            log_alphas[t][s] + np.log(self.null_arcs[s][s_y])
                        ])

        return log_alphas

    def backward(self, data, Q, init_beta=None):
        # Construct trellis for the forward pass with equalliy likely initial (stage-0) values

        betas_ = np.zeros((len(data) + 1, self.num_states))  # normalized betas

        if init_beta is None:
            betas_[-1] = [1] * self.num_states
        else:
            betas_[-1] = init_beta  # This should be an np array
        
        betas_[-1] = betas_[-1] / Q[-1]

        for t in range(len(data) - 1, -1, -1):
            # Calculate beta values for each state in this stage
            obs = data[t]  # Note: beta[t] actually means the prob of generating data[t:]
            betas_[t] = np.dot(betas_[t + 1], (self.transitions * self.emissions[obs]).T)

            # null arcs
            for s in reversed(self.topo_order):
                if s not in self.null_arcs:
                    continue
                for s_y in self.null_arcs[s]:
                    betas_[t][s] += betas_[t][s_y] * self.null_arcs[s][s_y]

            betas_[t] /= Q[t]

        # print("betas", betas)
        return betas_

    def backward_log(self, data, init_log_beta=None):
        # Construct trellis for the forward pass with equalliy likely initial (stage-0) values

        log_betas = np.empty((len(data) + 1, self.num_states))

        if init_log_beta is None:
            log_betas[-1] = np.zeros(self.num_states)
        else:
            log_betas[-1] = init_log_beta

        for t in range(len(data) - 1, -1, -1):
            # Calculate beta values for each state in this stage
            obs = data[t]
            for j in range(self.num_states):
                log_betas[t][j] = logsumexp(log_betas[t + 1] + np.log(self.emissions[obs][j]) + np.log(self.transitions[j]))

            # null arcs
            for s in reversed(self.topo_order):
                if s not in self.null_arcs:
                    continue
                for s_y in self.null_arcs[s]:
                    log_betas[t][s] = logsumexp([
                        log_betas[t][s],
                        log_betas[t][s_y] + np.log(self.null_arcs[s][s_y])
                    ])

        # print("betas", betas)
        return log_betas

    def un_norm_alphas_(self, alphas_, Q):
        alphas = np.copy(alphas_)
        cur_q = 1
        for t in range(alphas.shape[0]):
            cur_q *= Q[t]
            alphas[t] *= cur_q
        return alphas

    def un_norm_betas_(self, betas_, Q):
        betas = np.copy(betas_)
        cur_q = 1
        for t in range(betas.shape[0] - 2, -1, -1):
            cur_q *= Q[t + 1]
            betas[t] *= cur_q
        return betas

    def forward_backward(self, train, init_prob=None, init_beta=None, update_params=True):
        # Perform forward and backward passes to calculate alpha and beta values

        alphas_, Q = self.forward(train, init_prob=init_prob)
        betas_ = self.backward(train, Q, init_beta=init_beta)

        # short_thres = 10
        # if len(train) < short_thres:
        #     alphas = self.un_norm_alphas_(alphas_, Q)
        #     betas = self.un_norm_betas_(betas_, Q)
        #
        # log_alphas = self.forward_log(train, init_log_prob=(np.log(init_prob) if init_prob is not None else None))
        # log_betas = self.backward_log(train, init_log_beta=(np.log(init_beta) if init_beta is not None else None))

        # check (log_)alphas
        # for t in range(0, len(train)):
        #     if len(train) < short_thres:
        #         assert np.allclose(log_alphas[t], np.log(alphas)[t])
        #     assert np.allclose(log_alphas[t], (np.log(alphas_[t]) + np.log(Q[0:t + 1]).sum()), equal_nan=True)
        # assert np.log(Q).sum() - logsumexp(log_alphas[-1]) < 1e-6  # check likelihood with alphas

        # check (log_)betas
        # if len(train) < short_thres:
        #     for t in range(0, len(train) - 1):
        #         assert np.allclose(log_betas[t], np.log(betas)[t])
        #         assert np.allclose(log_betas[t], (np.log(betas_[t]) + np.log(Q[t + 1:]).sum()))
        # assert np.log(Q).sum() - logsumexp(log_betas[0] + np.log(self.emissions[:, train[0]])) < 1e-6   # check likelihood with betas

        # check posterior on states
        # for t in range(1, len(train) + 1):
        #     # At every time step, this is a posterior prob of passing through this state
        #     assert (np.dot(alphas_[t - 1], self.transitions * self.emissions[train[t - 1]]) * betas_[t]).sum() - 1 < 1e-6

        # check P(train)
        # assert logsumexp((log_alphas + log_betas)[0]) - logsumexp((log_alphas + log_betas)[-1]) < 1e-6
        # assert logsumexp((log_alphas + log_betas)[0]) - np.log(Q).sum() < 1e-6
        # assert logsumexp((log_alphas + log_betas)[0]) - (np.log(alphas_[0][0] * betas_[0][0]) + np.log(Q).sum()) < 1e-6

        # get the counts based on non-null arc posteriors P_{t}^{*}(arc)
        self.reset_counters()
        for t in range(1, len(train) + 1):
            obs = train[t - 1]
            step1 = alphas_[t - 1] * (self.transitions * self.emissions[obs]).T
            step2 = betas_[t]
            step3 = (step1.T * step2)
            assert np.sum(step3) - 1 < 1e-6
            self.output_arc_counts[obs] += step3

        # a dict of d[ix][iy]=c, where ix->iy is a null arc with count c
        for t in range(1, len(train)):  # no null transition on first and last step
            for ix in self.null_arcs.keys():
                for iy in self.null_arcs[ix].keys():
                    p = alphas_[t][ix] * self.null_arcs[ix][iy] * betas_[t][iy] * Q[t]
                    self.output_arc_counts_null[ix][iy] += p

        if update_params:
            self.update_params()

        return alphas_, betas_, Q

    def reset_counters(self):
        self.output_arc_counts = np.zeros((self.num_outputs, self.num_states, self.num_states))
        self.output_arc_counts_null = defaultdict(lambda: defaultdict(lambda: 0))

    def set_counters(self, another_output_arc_counts, another_output_arc_counts_null):
        self.output_arc_counts += another_output_arc_counts

        #for ix in another_output_arc_counts_null.keys():
        #    for iy in another_output_arc_counts_null[ix].keys():
        #        self.output_arc_counts_null[ix][iy] += another_output_arc_counts_null[ix][iy]

    def update_params(self):
        self.emissions = self.output_arc_counts / self.output_arc_counts.sum(axis=0)
        np.nan_to_num(self.emissions, copy=False, nan=0.0)
        self._assert_emission_probs()

        trans_sum = self.output_arc_counts.sum(axis=0)
        trans_sum_row = trans_sum.sum(axis=1)
        trans_new = np.zeros_like(self.transitions, dtype=np.float64)
        for index, x in np.ndenumerate(self.transitions):
            ix, iy = index
            trans_new[index] = trans_sum[index] / (trans_sum_row[ix] + sum(self.output_arc_counts_null[ix].values()))
        self.transitions = trans_new

        for ix in self.null_arcs.keys():
            for iy in self.null_arcs[ix].keys():
                self.null_arcs[ix][iy] = \
                    self.output_arc_counts_null[ix][iy] / (trans_sum_row[ix] + sum(self.output_arc_counts_null[ix].values()))

        self._assert_transition_probs()

    def log_likelihood(self, alphas_, betas_, Q):
        return np.log((alphas_[-1] * betas_[-1] * Q[-1]).sum()) + np.log(Q).sum()

    def compute_log_likelihood(self, data, init_prob, init_beta):
        alphas_, Q = self.forward(data, init_prob=init_prob)
        return np.log((alphas_[len(data)] * init_beta).sum()) + np.log(Q).sum()


if __name__ == "__main__":
    # HW3 as test case
    h = HMM(num_states=3, num_outputs=2)
    h.init_transition_probs(np.asarray([[1.0/2, 1.0/6, 1.0/6], [0, 0, 1.0/3], [3.0/4, 1.0/4, 0]], dtype=np.float64))

    emission_init_matrix = [[[1, 0.5, 1], [0, 0, 1.0/3], [0, 0, 0]], [[0, 0.5, 0], [1, 1, 2.0/3], [1, 1, 1]]]
    h.init_emission_probs(np.asarray(emission_init_matrix, dtype=np.float64))

    null_arcs = defaultdict(dict)
    null_arcs[0][2] = 1.0 / 6
    null_arcs[1][0] = 1.0 / 3
    null_arcs[1][2] = 1.0 / 3
    h.init_null_arcs(null_arcs)

    h.forward_backward([0, 1, 1, 0], init_prob=np.asarray([1, 0, 0]), update_params=False)
    log_likelihood = h.compute_log_likelihood([0, 1, 1, 0], init_prob=np.asarray([1, 0, 0]), init_beta=np.asarray([1, 1, 1]))
    print("log_likelihood", log_likelihood)
    #print("transitions", h.transitions)
    #print("null_arcs", h.null_arcs)
    #print("emissions", h.emissions)
