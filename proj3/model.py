import torch
import torch.nn as nn   
import torch.nn.functional as F

class LSTM_ASR(torch.nn.Module):
    def __init__(self, feature_type="discrete", input_size=64, hidden_size=256, num_layers=2,
                 output_size=28):
        super().__init__()
        assert feature_type in ['discrete', 'mfcc']
        # Build your own neural network. Play with different hyper-parameters and architectures.
        # === write your code here ===
        self.embeddings = nn.Embedding(256, input_size)
        if feature_type == 'mfcc':
            self.LSTM = torch.nn.LSTM(input_size=40, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.LSTM = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.type = feature_type

    def forward(self, batch_features):
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """
        # === write your code here ===

        if self.type == 'discrete':
            batch_features = self.embeddings(batch_features)
        output, _ = self.LSTM(batch_features)
        output = self.linear(output)
        # softmax the scores
        output = F.log_softmax(output, dim=2)
        return output
    