import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.1):
        self.dropout_rate = dropout_rate
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel,32,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(32,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.l_c1=nn.Linear(256,n_outputs)
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(self.call_bn(self.bn1, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=self.c2(h)
        h=F.leaky_relu(self.call_bn(self.bn2, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=self.c3(h)
        h=F.leaky_relu(self.call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=self.c4(h)
        h=F.leaky_relu(self.call_bn(self.bn4, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        return logit

    @staticmethod
    def call_bn(bn, x):
        return bn(x)
