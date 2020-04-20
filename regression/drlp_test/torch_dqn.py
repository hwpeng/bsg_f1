import torch
import torch.nn as nn
import torch.nn.functional as F

class torch_dqn(nn.Module):
    def __init__(self, name):
        super(torch_dqn, self).__init__()
        torch.manual_seed(2020)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 18)
        self.fc2 = nn.Linear(18, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_conv1_w(self):
        return self.conv1.weight.data.numpy()
    def get_conv1_b(self):
        return self.conv1.bias.data.numpy()
    def get_conv2_w(self):
        return self.conv2.weight.data.numpy()
    def get_conv2_b(self):
        return self.conv2.bias.data.numpy()
    def get_conv3_w(self):
        return self.conv3.weight.data.numpy()
    def get_conv3_b(self):
        return self.conv3.bias.data.numpy()
    def get_fc1_w(self):
        return self.fc1.weight.data.numpy()
    def get_fc1_b(self):
        return self.fc1.bias.data.numpy()
    def get_fc2_w(self):
        return self.fc2.weight.data.numpy()
    def get_fc2_b(self):
        return self.fc2.bias.data.numpy()


