# Frankenstein model
import torch 
from torch import nn, optim
import sys, math, random 
import numpy as np
from string import punctuation

'''
Instead of taking in characters like AliceModel, this model 
takes in words, hoping to create a more coherent output than
the Alice model could. It uses Embeddings to do this.
'''

class FrankenModel(nn.Module):

    def __init__(self, vocab_size, input_size, hidden_size, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, states):
        out = self.embed(x)
        out, states = self.lstm(out, states)
        out = torch.reshape(out, (-1, self.hidden_size))
        out = self.linear(out)
        # not to self: out is non-normalised probabilities, not one-hot
        return out, states 

    def init_states(self, batch_size):
        hidden = next(self.parameters()).data.new(
            self.n_layers, batch_size, self.hidden_size
        ).zero_()
        cell = next(self.parameters()).data.new(
            self.n_layers, batch_size, self.hidden_size
        ).zero_()
        states = (hidden, cell)
        return states 

file_path = "learning_pytorch\\Frankenstein\\frankenstein.txt"
data : list[str] = []
unique_words = []
word_to_idx = {}

with open(file_path, encoding='utf-8', errors='replace') as frank:
    raw_data = frank.read()
    for p in punctuation: 
        raw_data = raw_data.replace(p, "")
    for p in ['\u201c', "\u201d", "-"]:
        raw_data = raw_data.replace(p, "")
    data = raw_data.split()
    for i in range(len(data)):
        data[i] = data[i].lower().strip()
        if data[i] in unique_words: continue 
        unique_words.append(data[i])

    for i in range(len(unique_words)):
        word_to_idx[unique_words[i]] = i
    # len(unique_words = 7227)

def embed_encode(batch : list[str]) -> list[int]:
    return [word_to_idx[w] for w in batch]

def onehotencode(batch : list[str]):
    output = []
    for word in batch:
        output.append([0 if word != unique_words[i] else 1 for i in range(len(unique_words))])
    return output 

model = FrankenModel(len(unique_words), 100, 256, 2)

# batch information 
batch_size = 100
seq_len = 50
num_batches = int(len(data) / (batch_size * seq_len))

total_length = num_batches * (batch_size * seq_len)

loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)
epochs = 7

save_to_path = "learning_pytorch\\Frankenstein\\franken_model.pth"

count = 0
def main():
    # train the model 
    for e in range(epochs):
        print(f"Epoch {e+1}:")

        # create the batches 
        for i in range(num_batches):
            bs = batch_size * seq_len
            x_batch = data[i*bs:(i+1)*bs]
            y_batch = data[i*bs+1:min(total_length, (i+1)*bs+1)]
            if len(y_batch) < bs: y_batch.append(" ")
            assert len(y_batch) == len(x_batch)
            states = model.init_states(batch_size)

            # divide this batch into sequences
            x_seqs = []
            y_seqs = []
            for k in range(batch_size):
                x_seqs.append(embed_encode(x_batch[k*seq_len:(k+1)*seq_len]))
                y_seqs.append(embed_encode(x_batch[k*seq_len:(k+1)*seq_len]))
            
            x_seqs = torch.tensor(x_seqs)
            y_seqs = torch.tensor(y_seqs).reshape((batch_size*seq_len,)) 

            torch.autograd.set_detect_anomaly(True)

            y_pred, states = model(x_seqs, states)
            loss = loss_fn(y_pred, y_seqs)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            global count
            if (count % 5 == 0): print(f"Loss: {loss}")
            count += 1

    # save the model 
    torch.save(model.state_dict(), save_to_path)
if __name__ == '__main__':
    main()

