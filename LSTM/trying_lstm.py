import torch
from torch import nn, optim 
import random 
import numpy as np

# Define the Model 
class AliceModel(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.model = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.linlayer = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, states):

        out, states = self.model(x, states)
        out = torch.reshape(out, (-1, self.hidden_size))
        out = self.linlayer(out)
        return out, states 

    def init_states(self, batch_size):
        '''
        Reason: setting states = None at the start of each epoch
        causes error
        '''
        hidden = next(self.parameters()).data.new(
            self.n_layers, batch_size, self.hidden_size
        ).zero_()
        cell = next(self.parameters()).data.new(
            self.n_layers, batch_size, self.hidden_size
        ).zero_()
        states = (hidden, cell)
        return states 

# One hot encoding
def onehotencode(batch : str):
    output = []
    for c in batch: 
        output.append(
        [0 if c != unique_chars[i] else 1 for i in range(len(unique_chars))]
    )
    
    return output
    

# Read the file to train the model
data = ""
unique_chars = []
char_dict = {}
file_path = "learning_pytorch\\LSTM\\alice.txt"
with open(file_path, 'r') as alice:
    data = alice.read()
    for c in data:
        if not c: continue
        if c in unique_chars: continue 
        unique_chars.append(c)
    
    unique_chars.sort()
    for i in range(len(unique_chars)):
        char_dict[unique_chars[i]] = i
    
    alice.close()

# Determine number of batches 
seq_len = 75
batch_size = 100
n_batches = int(len(data) / (batch_size * seq_len))

total_length = n_batches * (batch_size * seq_len)

model = AliceModel(len(unique_chars), 256, 2)
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)
epochs = 35

# to save the model's parameters later
save_to_path = "learning_pytorch\\LSTM\\lstm_model.pth"

def main():
    # train the model 
    for e in range(epochs):
        model.train()
        print(f"Epoch: {e+1}")
        # states = None

        # input shape: (batch_size, seq_len, input_size)
        # output shape: (batch_size, seq_len, hidden_size) 

        # divide data into batches
        for i in range(n_batches):
            bs = batch_size * seq_len
            x_batch = data[i*bs:(i+1)*bs]
            y_batch = data[i*bs+1:max(total_length, (i+1)*bs+1)]
            if len(y_batch) < bs: y_batch += " "
            states = model.init_states(batch_size)

            # Detach hidden states to prevent backprop through entire history
            '''if states is not None:
                if isinstance(states, tuple):
                    states = tuple(s.detach() for s in states)
                else:
                    states = states.detach()'''

            # divide batch into sequences for times steps
            x_seqs = [] 
            y_seqs = []
            for k in range(batch_size): 
                # onehot encode each sequence 
                x_seqs.append(
                    onehotencode(x_batch[k*seq_len:(k+1)*seq_len])
                )
                y_seqs.append(
                    [char_dict[c] for c in y_batch[k*seq_len:(k+1)*seq_len]]
                )

            
            # convert the batch into a tensor 
            # shape of x_seq : (batch_size, seq_len, input_size)
            # shape of y_seq : (batch_size*seq_len, 1)
            
            x_seqs = torch.tensor(x_seqs, dtype=torch.float32)
            y_seqs = torch.tensor(y_seqs).reshape((batch_size*seq_len,))

            
            torch.autograd.set_detect_anomaly(True)

            y_pred, states = model(x_seqs, states)
            loss = loss_fn(y_pred, y_seqs)
            optimiser.zero_grad()
            loss.backward(retain_graph=True)
            optimiser.step()

            if random.randint(1,20) == 1: print(f"Loss: {loss}")

    # Save the model !! 

    torch.save(model.state_dict(), save_to_path)

if __name__ == '__main__': main()
