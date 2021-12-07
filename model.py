import torch.nn as nn
import torch


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
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )

        self.fc_block = nn.Sequential(
            nn.Linear(1600, 1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(1024, num_classes)
        )

        self.final = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.reshape(-1, 1600)
        x = self.fc_block(x)
        x = self.final(x)
        return x


def mnist_block(in_ch, out_ch):

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_ch),
        nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_ch)
    )


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.block1 = mnist_block(1, 32)
        self.block2 = mnist_block(32, 64)
        self.block3 = mnist_block(64, 128)
        self.block4 = mnist_block(128, 64)
        self.block5 = mnist_block(64, 32)

        self.conv_t1 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
        self.conv_t2 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_1 = self.block1(x)
        x = self.max_pool(x_1)
        x_2 = self.block2(x)
        x = self.max_pool(x_2)
        x = self.block3(x)

        x = self.conv_t1(x)
        x = torch.cat((x_2, x), dim=1)
        x = self.block4(x)
        x = self.conv_t2(x)
        x = torch.cat((x_1, x), dim=1)
        x = self.block5(x)
        x = self.final(x)

        return x


class ClassificationMnist(nn.Module):
    def __init__(self):
        super(ClassificationMnist, self).__init__()

        self.conv = nn.Sequential(
            mnist_block(128, 256),
            mnist_block(256, 128),
            mnist_block(128, 128)  # here 128, 7, 7
        )

        self.fc = nn.Linear(6272, 10)

        self.final = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, 6272)
        x = self.fc(x)
        return self.final(x)
