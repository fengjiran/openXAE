import torch
import torch.nn as nn


class Adder(nn.Module):
    def forward(self, x, y):
        return x + y


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.param = nn.Parameter(torch.rand(3, 4))
        self.linear = nn.Linear(4, 5)
        print(self.linear.weight.size())

    def forward(self, x):
        return torch.topk(torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3)


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(5, 4)
    a = net(x)

    # export torchscript
    with torch.no_grad():
        mod = torch.jit.trace(net, x)
        mod.save('test_inline_block.pt')
        print(mod.graph)


if __name__ == '__main__':
    test()
