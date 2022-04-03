from rnn import RNN
from brnn import BRNN
import torch.nn as nn

class Sequence(nn.Module):
    def __init__(self, **kwargs):
        super(Sequence, self).__init__()
        self.mode = kwargs.get("mode", "mtm")
        self.layers = nn.ModuleList(kwargs.get("layers", None ))  
        self.input_dim = kwargs.get("input_dim", 1)
        self.output_dim = kwargs.get("output_dim", 1)
        self.num_layers = kwargs.get("num_layers", 1)
        self.unit_output_dim = kwargs.get("unit_output_dim", self.input_dim)
        self.time_steps = kwargs.get("time_steps", self.output_dim)
        self.act_dim = kwargs.get("act_dim", self.input_dim)
        self.act = kwargs.get("act", nn.Identity())
        self.acty = kwargs.get("acty", nn.Identity())
        self.bidirectional = kwargs.get("bidirectional", True)
        self.device = kwargs.get("device", "cpu")
        
        if not self.layers:
            layers = []
            for _ in range(self.num_layers):
                if self.bidirectional:
                    layers.append(BRNN(input_dim = self.input_dim, output_dim = self.output_dim, num_layers = self.num_layers, act = self.act, acty = self.acty, device=self.device))
                else:
                    layers.append(RNN(input_dim = self.input_dim, output_dim = self.output_dim, num_layers = self.num_layers, act = self.act, acty = self.acty, device = self.device))
            self.layers = nn.ModuleList(layers)
        
    def forward(self,  X): 
        if self.mode == "otm": 
            assert(X.shape[1] == 1)
            X = torch.squeeze(X,1) 
            output_dim = self.layers[0].output_dim
            Y = torch.zeros(X.shape[0], output_dim, X.shape[1]).to(self.device)
            
            for layer in self.layers:
                assert(output_dim == layer.output_dim)
            
            for idx in range(output_dim):
                activations = torch.zeros(len(self.layers),X.shape[0],self.layers[0].act_dim).to(self.device)
                for i, layer in enumerate(self.layers): 
                    a,X = layer.models[idx].forward(activations[i, :, :], X) 
                    activations[i,:, :] = a
                    
                Y[:, idx, :] = X
                    
            return Y
             
        else: 
            for layer in self.layers:
                X = layer(X)
            return X
    
    