import torch
import torch.nn as nn
from cell import Cell

class RNN(nn.Module):
    def __init__(self, **kwargs):
        super(RNN, self).__init__()
        # output shape is 3d. n x output_dim x embed_length
        # input_dim is the embedding length(ie. word embedding length)
        
        self.input_dim = kwargs.get("input_dim", 1)
        self.act_dim = kwargs.get("act_dim", self.input_dim)
        self.output_dim = kwargs.get("output_dim", 1)
        self.time_steps = kwargs.get("time_steps", self.output_dim) 
        self.unit_output_dim = kwargs.get("unit_output_dim" , self.input_dim) 
        self.act = kwargs.get("act", nn.Identity())
        self.acty = kwargs.get("acty", nn.Identity())
        self.return_sequences = kwargs.get("return_sequences", False)
        self.device = kwargs.get("device", "cpu")
        
        assert(self.output_dim <= self.time_steps)
        # Populate the layer with the cells based on timesteps.
        self.models = nn.ModuleList([
            Cell(self.input_dim, self.unit_output_dim, act_dim = self.act_dim, act = self.act, acty = self.acty, device = self.device) 
        ] * self.time_steps )
        
    def forward(self,  X):
        # x is n x time_steps x embed_length 
        n = X.shape[0] 
        
        # make sure X axis 1 is less than time_steps
        assert(X.shape[1] <= self.time_steps)
        
        # Model expects inputs can have varying sizes and expects these inputs to be post-padded with NaN values.
        # non_padded_length aims to get the non-padded length of the input sequence of the particular example.
        non_padded_length = 0
        for i in range(X.shape[1]):
            if X[:, i, :].isnan().any().item(): 
                break
            non_padded_length += 1
        
        # Add NaN post-padding when input length less than time-steps
        pad_length = self.time_steps - X.shape[1]
        X = torch.cat((X, float('nan') + torch.zeros(n, pad_length, self.input_dim).to(self.device)), axis = 1).to(self.device)
        
        # Initialize the first activation.
        a = torch.zeros(n, self.act_dim).to(self.device)
        
        # Create the activations array
        A = torch.zeros(n, self.time_steps, self.act_dim).to(self.device)
        
        # Create the Y array, the individual y-predictions will be stored here
        Y = torch.zeros(n, self.time_steps, self.unit_output_dim).to(self.device)
        
        for i,cell in enumerate(self.models):
            # Get the input
            x = X[:, i, :] 
            
            # if input is padding, simply copy over previous activations and y
            if i >= non_padded_length:
                a,Y[:, i, :] = a, Y[:, i-1, :]  
                
            # if input is not padding, pass to rnn cell to get a and y
            else: 
                a,Y[:, i, :] = cell(a, x)   
                
            A[:, i, :] = a
            
        if self.return_sequences:
            return A
        
        else:
            # only return the last predictions detailed in output_dim
            if non_padded_length - self.output_dim >= 0:
                return Y[:, non_padded_length - self.output_dim:non_padded_length, :]
            
            else:
                return Y[:, :self.output_dim:, :]
                