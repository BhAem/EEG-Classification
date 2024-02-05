# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/20 13:41
import torch
from torch import nn


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv1d(channel, channel // 8, 1, padding=0, bias=True),
                nn.PReLU(),
                nn.Conv1d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.ca = nn.Sequential(
                nn.Conv1d(channel, channel // 8, 1, padding=0, bias=False),
                nn.PReLU(),
                nn.Conv1d(channel // 8, channel, 1, padding=0, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # 2 16 256 -> 2 16 1
        y = self.ca(y)  # 2 16 1 -> 2 16 1
        return x * y


class ESNet(nn.Module):

    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        return out[1:]

    def __init__(self, num_channels, T, num_classes):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5
        self.F = [num_channels * 2] + [num_channels * 4]
        self.K = 10
        self.S = 2

        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=num_channels * 2, kernel_size=(num_channels, 1))
        self.bn_0 = nn.BatchNorm2d(num_features=num_channels * 2)
        self.relu_0 = nn.PReLU()
        self.dropout_0 = nn.Dropout(self.dropout_level)
        self.lstm_0 = nn.LSTM(input_size=num_channels * 2, hidden_size=num_channels * 2, bidirectional=False, num_layers=1)

        self.conv_1 = nn.Conv2d(in_channels=self.F[0], out_channels=self.F[0], kernel_size=(1, self.K), stride=(1, self.S))
        self.bn_1 = nn.BatchNorm2d(num_features=self.F[0])
        self.relu_1 = nn.PReLU()
        self.dropout_1 = nn.Dropout(self.dropout_level)
        self.lstm_1 = nn.LSTM(input_size=self.F[0], hidden_size=self.F[0], bidirectional=False, num_layers=1)

        self.flatten = nn.Flatten()

        self.fcUnit = 1984  # 1984 960 15936

        self.dense_layers = nn.Sequential(
            nn.Linear(self.fcUnit, self.fcUnit // 10),
            nn.PReLU(),
            nn.Linear(self.fcUnit // 10, num_classes))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.calayer = CALayer(16)
        self.calayer2 = CALayer(16)
        self.palayer = PALayer(16)
        self.palayer2 = PALayer(16)

    def forward(self, x, flag=True):
        out = self.conv_0(x)
        out = self.bn_0(out)
        out = self.relu_0(out)
        out = self.dropout_0(out)
        out = out.squeeze(2)
        b, c, T = out.size()
        out = out.view(out.size(-1), -1, c)
        out, (h0, c0) = self.lstm_0(out)
        T, b, c = out.size()
        out = out.view(b, c, T)
        # out = out.unsqueeze(2)

        # res = self.avg_pool(out)
        out = self.calayer(out)
        out = self.palayer(out)
        out = out.unsqueeze(2)

        out = self.conv_1(out)
        out = self.bn_1(out)
        out = self.relu_1(out)
        out = self.dropout_1(out)
        out = out.squeeze(2)
        b, c, T = out.size()
        out = out.view(out.size(-1), -1, c)
        out, (h1, c1) = self.lstm_1(out, (h0, c0))

        # T, b, c = out.size()
        # out = out.view(b, c, T)
        # out = self.calayer2(out)
        # out = self.palayer2(out)

        # T, b, c = out.size()
        # out = out.view(b, c, T)
        # out = res * out

        out = out.view(b, T * c, -1)

        feature = self.flatten(out)
        out = self.dense_layers(feature)
        if flag:
            return out, feature
        else:
            return out


if __name__ == '__main__':

    net = ESNet(8, 256, 12)
    from thop import profile
    input = torch.randn(1, 1, 8, 256)
    flops, params = profile(net, inputs=(input,))
    print(flops)
    print(params)

    # net = ESNet(8, 2000, 4)
    # from thop import profile
    # input = torch.randn(10, 1, 8, 2000)
    # flops, params = profile(net, inputs=(input,))
    # print(flops)
    # print(params)
