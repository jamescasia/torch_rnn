import torch
import torch.nn as nn

class Cell(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Cell, self).__init__()
          
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_dim = kwargs.get("act_dim", self.input_dim)
        
        self.linear = nn.Linear(self.act_dim+self.input_dim,self.act_dim) 
        self.activation = kwargs.get("act", nn.Identity())
        self.linear_y = nn.Linear(self.act_dim, self.output_dim)
        self.activation_y = kwargs.get("acty", nn.Identity()) 
        self.device = kwargs.get("device", "cpu")
    
    def forward(self, a, X): 
        # X is n x input_dim
        # a is n x act_dim 
        assert(a.shape[1] == self.act_dim)
        # concatenate a and X since they will be transformed by the same Linear.
        X = torch.cat((a,X), axis = 1).to(self.device)
        # transform the inputs
        X = self.linear(X)
        a = self.activation(X)
        X = self.linear_y(a)
        Y = self.activation_y(X)
        return a,Y 