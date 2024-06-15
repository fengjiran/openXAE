import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(2, 4), stride=(2, 1), padding=2,
                               dilation=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=24, kernel_size=(1, 3), stride=1, padding=(2, 4),
                               dilation=1, groups=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=28, kernel_size=(5, 4), stride=1, padding='valid',
                               dilation=1, groups=4, bias=True)
        self.conv4 = nn.Conv2d(in_channels=28, out_channels=32, kernel_size=3, stride=1, padding='same',
                               dilation=(1, 2), groups=2, bias=False, padding_mode='zeros')
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=3, dilation=1,
                               groups=32, bias=True, padding_mode='reflect')
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=28, kernel_size=2, stride=1, padding=2, dilation=1,
                               groups=1, bias=False, padding_mode='replicate')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64, 64)
    a = net(x)

    # export torchscript
    with torch.no_grad():
        mod = torch.jit.trace(net, x)
        mod.save('test_nn_Conv2d.pt')


if __name__ == '__main__':
    test()
    # if test():
    #     exit(0)
    # else:
    #     exit(1)
