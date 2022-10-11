import os

import torch
from torch.autograd import Variable

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from .Attention import Attention
from torch.autograd import Variable
from modules.alpha_nn.modeling_base import AttentionSelf as Attention



class BasicModule(torch.nn.Module):

    def __init__(self, args):
        super(BasicModule,self).__init__()
        self.args = args
        self.model_name = str(type(self))

    def pad_doc(self,words_out,doc_lens):
        pad_dim = words_out.size(1)
        max_doc_len = max(doc_lens)
        sent_input = []
        start = 0
        for doc_len in doc_lens:
            stop = start + doc_len
            valid = words_out[start:stop]                                       # (doc_len,2*H)
            start = stop
            if doc_len == max_doc_len:
                sent_input.append(valid.unsqueeze(0))
            else:
                pad = Variable(torch.zeros(max_doc_len-doc_len,pad_dim))
                if self.args.device is not None:
                    pad = pad.cuda()
                sent_input.append(torch.cat([valid,pad]).unsqueeze(0))          # (1,max_len,2*H)
        sent_input = torch.cat(sent_input,dim=0)                                # (B,max_len,2*H)
        return sent_input
    
    def save(self):
        checkpoint = {'model':self.state_dict(), 'args': self.args}
        best_path = os.path.join(self.args.output,self.model_name+'.pt')
        #best_path = '%s%s_seed_%d.pt' % (self.args.output,self.model_name,self.args.seed)
        #best_path = '%s%s_seed_%d.pt' % (self.args.save_dir,self.model_name,self.args.seed)
        torch.save(checkpoint,best_path)

        return best_path

    def load(self, best_path):
        if self.args.device is not None:
            data = torch.load(best_path)['model']
        else:
            data = torch.load(best_path, map_location=lambda storage, loc: storage)['model']
        self.load_state_dict(data)
        if self.args.device is not None:
            return self.cuda()
        else:
            return self

class AttnRNN(BasicModule):
    def __init__(self, args, embed=None):
        super(AttnRNN,self).__init__(args)
        self.model_name = 'AttnRNN'
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num

        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V,P_D)
        self.rel_pos_embed = nn.Embedding(S,P_D)
        self.embed = nn.Embedding(V,D,padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.attn = Attention()
        self.word_query = nn.Parameter(torch.randn(1,1,2*H))
        self.sent_query = nn.Parameter(torch.randn(1,1,2*H))

        self.word_RNN = nn.GRU(
                        input_size = D,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        self.sent_RNN = nn.GRU(
                        input_size = 2*H,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
               
        self.fc = nn.Linear(2*H,2*H)

        # Parameters of Classification Layer
        self.content = nn.Linear(2*H,1,bias=False)
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
        self.abs_pos = nn.Linear(P_D,1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))
    def forward(self,x,doc_lens):
        N = x.size(0)
        L = x.size(1)
        B = len(doc_lens)
        H = self.args.hidden_size
        word_mask = torch.ones_like(x) - torch.sign(x)
        word_mask = word_mask.data.type(torch.cuda.ByteTensor).view(N,1,L)
        
        x = self.embed(x)                                # (N,L,D)
        x,_ = self.word_RNN(x)
        
        # attention
        query = self.word_query.expand(N,-1,-1).contiguous()
        self.attn.set_mask(word_mask)
        word_out = self.attn(query,x)[0].squeeze(1)      # (N,2*H)

        x = self.pad_doc(word_out,doc_lens)
        # sent level GRU
        sent_out = self.sent_RNN(x)[0]                                           # (B,max_doc_len,2*H)
        #docs = self.avg_pool1d(sent_out,doc_lens)                               # (B,2*H)
        max_doc_len = max(doc_lens)
        mask = torch.ones(B,max_doc_len)
        for i in range(B):
            for j in range(doc_lens[i]):
                mask[i][j] = 0
        sent_mask = mask.type(torch.cuda.ByteTensor).view(B,1,max_doc_len)
        
        # attention
        query = self.sent_query.expand(B,-1,-1).contiguous()
        self.attn.set_mask(sent_mask)
        docs = self.attn(query,x)[0].squeeze(1)      # (B,2*H)
        probs = []
        for index,doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index,:doc_len,:]                            # (doc_len,2*H)
            doc = F.tanh(self.fc(docs[index])).unsqueeze(0)
            s = Variable(torch.zeros(1,2*H))
            if self.args.device is not None:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)                                                # (1,2*H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device is not None:
                    abs_index = abs_index.cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)
                
                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device is not None:
                    rel_index = rel_index.cuda()
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                
                # classification layer
                content = self.content(h) 
                salience = self.salience(h,doc)
                novelty = -1 * self.novelty(h,F.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob,h)
                #print position,F.sigmoid(abs_p + rel_p)
                probs.append(prob)
        return torch.cat(probs).squeeze()

class CNN_RNN(BasicModule):
    def __init__(self, args, embed=None):
        super(CNN_RNN,self).__init__(args)
        self.model_name = 'CNN_RNN'
        self.args = args
        
        Ks = args.kernel_sizes
        Ci = args.embed_dim
        Co = args.kernel_num
        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num
        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V,P_D)
        self.rel_pos_embed = nn.Embedding(S,P_D)
        self.embed = nn.Embedding(V,D,padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.convs = nn.ModuleList([ nn.Sequential( nn.Conv1d(Ci,Co,K), nn.BatchNorm1d(Co), nn.LeakyReLU(inplace=True), nn.Conv1d(Co,Co,K), nn.BatchNorm1d(Co), nn.LeakyReLU(inplace=True)) for K in Ks])
        self.sent_RNN = nn.GRU( input_size = Co * len(Ks), hidden_size = H, batch_first = True, bidirectional = True)
        self.fc = nn.Sequential( nn.Linear(2*H,2*H), nn.BatchNorm1d(2*H), nn.Tanh())
        # Parameters of Classification Layer
        self.content = nn.Linear(2*H,1,bias=False)
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
        self.abs_pos = nn.Linear(P_D,1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))

    def max_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))
        out = torch.cat(out).squeeze(2)
        return out

    def avg_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.avg_pool1d(t,t.size(2)))
        out = torch.cat(out).squeeze(2)
        return out

    def forward(self,x,doc_lens):
        #sent_lens = torch.sum(torch.sign(x),dim=1).data 
        H = self.args.hidden_size
        x = self.embed(x)                                                       # (N,L,D)
        # word level GRU
        x = [conv(x.permute(0,2,1)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x,1)
        # make sent features(pad with zeros)
        x = self.pad_doc(x,doc_lens)

        # sent level GRU
        sent_out = self.sent_RNN(x)[0]                                           # (B,max_doc_len,2*H)
        docs = self.max_pool1d(sent_out,doc_lens)                                # (B,2*H)
        docs = self.fc(docs)
        probs = []
        for index,doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index,:doc_len,:]                            # (doc_len,2*H)
            doc = docs[index].unsqueeze(0)
            s = Variable(torch.zeros(1,2*H))
            if self.args.device is not None:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)                                                # (1,2*H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device is not None:
                    abs_index = abs_index.cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)
                
                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device is not None:
                    rel_index = rel_index.cuda()
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                
                # classification layer
                content = self.content(h) 
                salience = self.salience(h,doc)
                novelty = -1 * self.novelty(h,F.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                prob = F.sigmoid(content + self.bias)
                #prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob,h)
                probs.append(prob)
        return torch.cat(probs).squeeze()

class RNN_RNN(BasicModule):
    def __init__(self, args, embed=None):
        super(RNN_RNN, self).__init__(args)
        self.model_name = 'RNN_RNN'
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num
        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V,P_D)
        self.rel_pos_embed = nn.Embedding(S,P_D)
        self.embed = nn.Embedding(V,D,padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.word_RNN = nn.GRU(
                        input_size = D,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        self.sent_RNN = nn.GRU(
                        input_size = 2*H,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        self.fc = nn.Linear(2*H,2*H)

        # Parameters of Classification Layer
        self.content = nn.Linear(2*H,1,bias=False)
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
        self.abs_pos = nn.Linear(P_D,1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))

    def max_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out
    def avg_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.avg_pool1d(t,t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out
    def forward(self,x,doc_lens):
        sent_lens = torch.sum(torch.sign(x),dim=1).data 
        x = self.embed(x)                                                      # (N,L,D)
        # word level GRU
        H = self.args.hidden_size
        x = self.word_RNN(x)[0]                                                 # (N,2*H,L)
        #word_out = self.avg_pool1d(x,sent_lens)
        word_out = self.max_pool1d(x,sent_lens)
        # make sent features(pad with zeros)
        x = self.pad_doc(word_out,doc_lens)

        # sent level GRU
        sent_out = self.sent_RNN(x)[0]                                           # (B,max_doc_len,2*H)
        #docs = self.avg_pool1d(sent_out,doc_lens)                               # (B,2*H)
        docs = self.max_pool1d(sent_out,doc_lens)                                # (B,2*H)
        probs = []
        for index,doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index,:doc_len,:]                            # (doc_len,2*H)
            doc = F.tanh(self.fc(docs[index])).unsqueeze(0)
            s = Variable(torch.zeros(1,2*H))
            if self.args.device is not None:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)                                                # (1,2*H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device is not None:
                    abs_index = abs_index.cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)
                
                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device is not None:
                    rel_index = rel_index.cuda()
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                
                # classification layer
                content = self.content(h) 
                salience = self.salience(h,doc)
                novelty = -1 * self.novelty(h,F.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob,h)
                probs.append(prob)
        return torch.cat(probs).squeeze()
