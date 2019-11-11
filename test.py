import torch 
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m = nn.Linear(2,2,bias=False)
        self.a = nn.Parameter(torch.randn(3,5,requires_grad=True))

    def forward(x):
        #out = m(x)
        return 0

n = Net()
print(list(n.parameters()))