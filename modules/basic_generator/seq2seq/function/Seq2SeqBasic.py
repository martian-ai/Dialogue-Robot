import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, seq_input, hidden):
        embedded = self.embedding(seq_input).view(1, 1, self.hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return hidden

    def sample(self,seq_list):
        word_inds = torch.LongTensor(seq_list).to(device)
        h = self.initHidden()
        for word_tensor in word_inds:
            h = self(word_tensor,h)
        return h

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.maxlen = 10

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, seq_input, hidden):
        output = self.embedding(seq_input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def sample(self,pre_hidden):
        inputs = torch.tensor([SOS_token], device=device)
        hidden = pre_hidden
        res = [SOS_token]
        for i in range(self.maxlen):
            output,hidden = self(inputs,hidden)
            topv, topi = output.topk(1)
            if topi.item() == EOS_token:
                res.append(EOS_token)
                break
            else:
                res.append(topi.item())
            inputs = topi.squeeze().detach()
        return res


