from rnn import RNN
import torch
import torch.nn as nn
class BRNN(nn.Module):
    
    def __init__(self, **kwargs):
        super(BRNN, self).__init__()
        
        self.input_dim = kwargs.get("input_dim", 1)
        self.act_dim = kwargs.get("act_dim", self.input_dim)
        self.output_dim = kwargs.get("output_dim", 1)
        self.time_steps = kwargs.get("time_steps", self.output_dim) 
        self.unit_output_dim = kwargs.get("unit_output_dim" , self.input_dim)
        self.pad_seq = kwargs.get("pad_seq", "pre")
        self.act = kwargs.get("act", nn.Identity())
        self.acty = kwargs.get("acty", nn.Identity())  
        self.return_sequences = kwargs.get("return_sequences", False)
        self.device = kwargs.get("device", "cpu")
        
        # ensure output_dim less or equal to time steps.
        assert(self.output_dim <= self.time_steps)
        
        # Create the two RNN layers 
        self.sequence = RNN(input_dim = self.input_dim, act_dim = self.act_dim, time_steps = self.time_steps, output_dim = self.output_dim, unit_output_dim = self.unit_output_dim, return_sequences = True, act = self.act, acty = self.acty)
        
        self.reverse_sequence = RNN(input_dim = self.input_dim, act_dim = self.act_dim, time_steps = self.time_steps, output_dim = self.output_dim, unit_output_dim = self.unit_output_dim, return_sequences = True, act = self.act, acty = self.acty)
        # Populate linear layer for each time step.
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(self.act_dim*2,self.unit_output_dim), self.acty)]* self.time_steps)
        
    def forward(self, X):
        n = X.shape[0]
        # ensure X seq_length less than or equal to time_steps
        assert(X.shape[1] <= self.time_steps)
        
        # get activations from both rnns
        A1 = self.sequence(X)  
        A2 = self.reverse_sequence(torch.flip(X, dims = [1]))
        
        # initialize outputs and activations
        Y = torch.zeros(n, self.time_steps, self.unit_output_dim).to(self.device)
        A = torch.zeros(n,  self.time_steps, self.act_dim*2).to(self.device)
        
        # calculate activations 
        for i in range(self.time_steps): 
            linear = self.linears[i]
            a = torch.cat((A1[:, i, :], A2[:, i,:]), axis = 1)
            
            A[:, i, :] = a
            Y[:, i, :] = linear(a) 
            
        if self.return_sequences:
            return A[:, -self.output_dim:, :]
        else:
            return Y[:, -self.output_dim:,:] 
                     
         