import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(False),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels, num_hiddens // 8, kernel_size=4, stride=2, padding=1)  # 256 -> 128
        self._conv_2 = nn.Conv2d(num_hiddens // 8, num_hiddens // 4, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        self._conv_3 = nn.Conv2d(num_hiddens // 4, num_hiddens // 2, kernel_size=4, stride=2, padding=1)  # 64 -> 32
        self._conv_4 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)  # 32 -> 16
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, x):
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = F.relu(self._conv_3(x))
        x = F.relu(self._conv_4(x))
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2,padding=1)  # 16 -> 32
        self._conv_trans_2 = nn.ConvTranspose2d(num_hiddens // 2, num_hiddens // 4, kernel_size=4, stride=2,padding=1)  # 32 -> 64
        self._conv_trans_3 = nn.ConvTranspose2d(num_hiddens // 4, num_hiddens // 8, kernel_size=4, stride=2,padding=1)  # 64 -> 128
        self._conv_trans_4 = nn.ConvTranspose2d(num_hiddens // 8, 3, kernel_size=4, stride=2, padding=1)  # 128 -> 256

    def forward(self, x):
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        x = F.relu(self._conv_trans_2(x))
        x = F.relu(self._conv_trans_3(x))
        return self._conv_trans_4(x)