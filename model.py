from Encoder import WordEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class MUDE(nn.Module):
    def __init__(self, char_vocab_size, d_emb, h, n, d_hidden, vocab_size, dropout):
        super(MUDE, self).__init__()
        self.d_emb = d_emb
        self.emb_layer = nn.Embedding(char_vocab_size, d_emb)
        self.word_encoder = WordEncoder(N=n, d_model=d_emb, h=h, dropout=dropout)
        self.lstm = nn.LSTM(d_emb, d_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(d_hidden*2, vocab_size)

        self.seq_rnn = nn.GRU(d_emb, d_emb, batch_first=True)
        self.seq_pred = nn.Linear(d_emb, char_vocab_size)
    def forward(self, input, mask, hidden=None):
        """mask shape: batch_size, l, c, c
        input shape: batch_size, l, c
        """
        batch_size, l, c = input.shape
        x_char_emb = self.emb_layer(input).view(-1, c, self.d_emb) # batch_size, l, c, d_emb -> batch_size x l, c, d_emb
        x = self.word_encoder(x_char_emb, mask.view(-1, c, c)).view(batch_size, l, self.d_emb)
        
        seq_output, seq_hidden = self.seq_rnn(x_char_emb[:,:-1,:], x.view(1, -1, self.d_emb).contiguous()) # batch_size x l, c-1, d_emb
        seq_output = F.softmax(self.seq_pred(seq_output), dim=-1).view(batch_size, l, c-1, -1)
        if hidden != None:
            output, hidden= self.lstm(x, hidden)
        else:
            output, hidden= self.lstm(x)
        output = self.dropout(output)
        output = F.softmax(self.pred(output), dim=-1)
        return output, hidden, seq_output

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        # LSTM
        return (weight.new_zeros(1, bsz, self.d_hidden),
                    weight.new_zeros(1, bsz, self.d_hidden))
