import torch.nn as nn


def encoder_layer(n_channel):
    return nn.Sequential(
        nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), padding=1),
        nn.LeakyReLU()
    )


def decoder_layer(n_channel):
    return nn.Sequential(
        nn.ConvTranspose2d(n_channel, n_channel, kernel_size=(3, 3), padding=1),
        nn.LeakyReLU()
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

        self.conv_block = [encoder_layer(3) for _ in range(num_layers)]
        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        x = self.conv_block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, image_size):
        super(Decoder, self).__init__()

        self.conv_block = [decoder_layer(3) for _ in range(num_layers)]
        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        x = self.conv_block(x)

        return x


class ClassificationModel(nn.Module):
    def __init__(self, num_classes=100):
        super(ClassificationModel, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(5, 5), stride=(3, 3)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d()
        )

        self.fc_block = nn.Sequential(
            nn.Linear(1600, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.reshape(-1, 1600)
        x = self.fc_block(x)
        return x
