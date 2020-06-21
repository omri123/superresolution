import torch.nn as nn

def get_model():
    """

    :return: a model that gets an image and return image with the same size.
    """
    return SimpleModel()


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.block_a = ConvBlock(in_channel=3, out_channel=9, ker_size=3, padd=1, stride=1)
        self.block_b = ConvBlock(in_channel=9, out_channel=9, ker_size=3, padd=1, stride=1)
        self.conv_e = nn.Conv2d(9, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, img):
        h = self.block_a(img)
        h = self.block_b(h)
        h = self.conv_e(h)
        img = nn.Tanh()(h)
        return img
