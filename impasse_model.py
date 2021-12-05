import torch.nn as nn


def encoder_layer(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1),  # is not resized
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(),
        nn.Dropout()  # 0.5 default
    )


def decoder_layer(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1),  # is not resized
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(),
        nn.Dropout()  # 0.5 default
    )


def fc_layer(in_channel, out_channel):
    return nn.Sequential(
        nn.Linear(in_channel, out_channel),
        nn.BatchNorm1d(out_channel),
        nn.LeakyReLU(),
        nn.Dropout()
    )


class AECifar(nn.Module):
    def __init__(self, num_layers=2):
        super(AECifar, self).__init__()
        image_size = 32  # 32 it's size of cifar dataset
        self.encoder = Encoder(num_layers=num_layers, image_size=image_size)
        self.decoder = Decoder(num_layers=num_layers, image_size=image_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, image_size):
        super(Encoder, self).__init__()

        # conv layers from 3 to 2^(n+1) channel
        self.conv_block = [encoder_layer(3, 4)] + \
                          [encoder_layer(2 ** x, 2 ** (x + 1)) for x in range(2, num_layers + 1)]
        self.conv_block = nn.Sequential(*self.conv_block)

        input_size = image_size**2 * 2**(num_layers + 1)

        self.fc_block = []

        for n in range(0, num_layers - 1):
            x = int(input_size / 2 ** n)
            y = int(input_size / 2 ** (n + 1))
            self.fc_block.append(
                fc_layer(x, y)
            )

        x = int(input_size / 2 ** (num_layers - 1))
        y = int(input_size / 2 ** num_layers)
        self.fc_block.append(nn.Linear(x, y))

        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, image_size):
        super(Decoder, self).__init__()

        input_size = 2 * image_size**2

        self.image_size = image_size
        self.fc_block = []

        for n in range(0, num_layers):
            x = int(input_size * 2 ** n)
            y = int(input_size * 2 ** (n + 1))
            self.fc_block.append(
                fc_layer(x, y)
            )
        self.fc_block = nn.Sequential(*self.fc_block)

        # conv layers from 2^(n+1) to 3 channel
        self.conv_block = [decoder_layer(2 ** x, 2 ** (x - 1)) for x in range(num_layers + 1, 2, -1)] + \
                          [nn.Conv2d(4, 3, kernel_size=(3, 3), padding=1)]

        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        x = self.fc_block(x)
        x = x.reshape(x.shape[0], -1, self.image_size, self.image_size)
        x = self.conv_block(x)

        return x
