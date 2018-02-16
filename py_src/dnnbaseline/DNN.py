import torch.nn as nn



def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-0.05, 0.05)



class DNN(nn.Module):
    def __init__(self, inputdim, outputdim, activation='relu'):
        super(DNN, self).__init__()
        self.input = inputdim
        self.hidden = 2048
        self.output = outputdim
        self.layers = []
        self.activator = nn.Sigmoid() if activation == 'sigm' else nn.ReLU()
        self.layers.append(nn.BatchNorm1d(self.input))
        self.layers.append(nn.Linear(self.input, self.hidden))
        self.layers.append(self.activator)
        #self.layers.append(nn.Dropout())

        for i in range(1, 6):
            self.layers.append(nn.BatchNorm1d(self.hidden))
            self.layers.append(nn.Linear(self.hidden, self.hidden))
            self.layers.append(self.activator)
            #self.layers.append(nn.Dropout())
    
        self.layers.append(nn.BatchNorm1d(self.hidden))
        self.layers.append(nn.Linear(self.hidden, self.output))
        self.network = nn.Sequential(*self.layers)
        self.network.apply(init_weights)
    def forward(self, x):
        return self.network(x)



        '''bs, _ = x.size()
        for m in self.network:
            if bs==1 and type(m) == nn.BatchNorm1d:
                continue
            x = m(x)
        return x'''
