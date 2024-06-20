import torch
import torch.nn as nn


class Adder(nn.Module):
    def forward(self, x, y):
        return x + y


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.adder = Adder()

    def forward(self, x):
        x = x * 2
        x = self.adder(x, 2)
        x.add_(3)
        x = x.view(-1)
        if x[0] > 1:
            return x[0]
        else:
            return x[-1]


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64, 64)
    a = net(x)

    # export torchscript
    with torch.no_grad():
        mod = torch.jit.trace(net, x)
        mod.save('test_inline_block.pt')
        print(mod.graph)


if __name__ == '__main__':
    test()
