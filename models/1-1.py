import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Some Information about MLP"""

    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))


class MySequential(nn.Module):
    """Some Information about MySequential"""

    def __init__(self, *args):
        super(MySequential, self).__init__()
        for block in args:
            self._modules[block] = block             

    def forward(self, x):
        for block in self._modules.values():
            x = block(x) 
        return x


if __name__ == '__main__':
    x = torch.rand(2, 20)
    # net = MLP()
    # print(net(x))
    net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
    print(net(x))


if __name__ == '__main__':
    import requests
    for i in range(104):
        print(i)