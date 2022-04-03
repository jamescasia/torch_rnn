from torch_rnn import Cell, RNN, BRNN, Sequence
from torch.optim import Adam
import torch
import torch.nn as nn 
embed_length = 1
n = 1000
seq_length = 3
output_dim = 5
 
torch.manual_seed(699)
seq = Sequence(input_dim = embed_length, output_dim = output_dim, 
               num_layers = 1,mode="mtm", bidirectional = False, device="cpu" )
# seq = BRNN(input_dim = embed_length, output_dim = output_dim,  
#           device="cpu"  )
# seq =  RNN(input_size = embed_length,hidden_size = embed_length, bidirectional = True)
X = torch.zeros(n, seq_length, embed_length)
Y = torch.zeros(n, output_dim, embed_length)

for _ in range(n):
    X[_, :, :] = torch.zeros(seq_length, embed_length) + _
    # X[_,-1, :] = torch.randint(  1000, (1,embed_length))
    Y[_, :, :] = torch.zeros(output_dim, embed_length) + _  
    # Y[_,1, :] =  X[_,-1, :]
    
print("X")
print(X)
print("Y")
print(Y)
print("Training")
 
adam = Adam(seq.parameters() , lr = 1e-1, weight_decay = 1e-5 )
epochs = 5000

for _ in range(epochs):
    
    adam.zero_grad()
    
    Y_hat = seq(X) 
    
    loss = ((Y - Y_hat)**2).mean()
    
    loss.backward()
    
    adam.step()
    if epochs - _ < 10:
        print("loss:",loss.item())

print(seq(X))