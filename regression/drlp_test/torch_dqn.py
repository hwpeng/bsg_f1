import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym_env import *
import struct

def input_process(x):
    z = np.zeros((1, 4, 84, 84))
    for d in range(4):
        for r in range(84):
            for c in range(84):
                z[0, d, r, c] = x[d*84*84+r*84+c]
    z = torch.from_numpy(z).type('torch.FloatTensor')
    return z

def save_grad(dqn, name):
    def hook(grad):
        dqn.grads[name] = grad
    return hook

class torch_dqn(nn.Module):
    def __init__(self, name):
        super(torch_dqn, self).__init__()
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(2020)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 4)
        self.grads = {}
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_relu = F.relu(x1)

        x2 = self.conv2(x1_relu)
        x2_relu = F.relu(x2)

        x3 = self.conv3(x2_relu)
        # This is how DRLP reshape activations before relu
        x3 = x3.permute(0,3,2,1)
        x3 = x3.reshape(-1, 3136)
        x3.register_hook(save_grad(self, 'x3'))
        x3_relu = F.relu(x3)

        x4 = self.fc1(x3_relu)
        x4.register_hook(save_grad(self, 'x4'))
        x4_relu = F.relu(x4)

        x5 = self.fc2(x4_relu)
        return x5

    def forward_no_grad(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))

        # This is how DRLP reshape activations before relu
        x3 = x3.permute(0,3,2,1)
        x3 = x3.reshape(-1, 3136)

        x4 = F.relu(self.fc1(x3))
        x5 = self.fc2(x4)
        return x5


    def c_call_forward(self, x):
        x = input_process(x)
        x = self.forward(x)
        x = x.data.numpy()
        return np.squeeze(x)

    def c_call_train(self, state, next_state, reward, action, done, gamma):
        state = input_process(state)
        state_values = self.forward(state)
        with torch.no_grad():
            next_state = input_process(next_state)
            next_values = self.forward_no_grad(next_state)
            next_max_index = next_values.argmax(1).data[0]
            if (done == 0):
                target = reward + gamma*next_values[0, next_max_index]
            else:
                target = reward

        input = state_values[0, action]
        loss = F.mse_loss(input, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #  torch.set_printoptions(profile="full")
        #  print(self.grads['x3'])
        #  torch.set_printoptions(profile="default")
        return np.array([input.data, target.data, loss.data])

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
     
    def get_conv1_dw(self):
        return self.conv1.weight.grad.data.numpy()
    def get_conv1_db(self):
        return self.conv1.bias.grad.data.numpy()
    def get_conv2_dw(self):
        return self.conv2.weight.grad.data.numpy()
    def get_conv2_db(self):
        return self.conv2.bias.grad.data.numpy()
    def get_conv3_dw(self):
        return self.conv3.weight.grad.data.numpy()
    def get_conv3_db(self):
        return self.conv3.bias.grad.data.numpy()
    def get_fc1_dw(self):
        return self.fc1.weight.grad.data.numpy()
    def get_fc1_db(self):
        return self.fc1.bias.grad.data.numpy()
    def get_fc2_dw(self):
        return self.fc2.weight.grad.data.numpy()
    def get_fc2_db(self):
        return self.fc2.bias.grad.data.numpy()

    # For testing!!!
    def conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0,3,2,1)
        x = x.reshape(-1, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train(self, state, next_state, reward, action, done, gamma):
        state = torch.from_numpy(state)
        state.unsqueeze_(0)
        state_values = self.forward(state)
        next_state = torch.from_numpy(next_state)
        next_state.unsqueeze_(0)
        next_values = self.forward(next_state)
        next_max_index = next_values.argmax(1).data[0]
        if (done == 0):
            target = reward + gamma*next_values[0, next_max_index]
        else:
            target = reward
        input = state_values[0, action]

        loss = F.mse_loss(input, target)
        self.zero_grad()
        loss.backward()

def f2b(num):
    a = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))
    return hex(int(a,2))

dqn = torch_dqn('ad')
env = gym_env('Breakout-v0')
px = env.reset()
state = np.copy(px)
cx, reward, done, __ = env.env.step(3)
cx = process_frame84(cx)
px[3,:,:] = cx
next_state = np.copy(px)

# dqn.train(state, next_state, reward, 3, done, 0.95)
# print(dqn.get_conv1_db().shape)

# o = dqn.conv_forward(x).data.numpy()
# w3 = dqn.conv3.weight.data.numpy()
# w2 = dqn.conv2.weight.data.numpy()
w2 = dqn.fc2.weight.data.numpy()

a = dqn.c_call_train(np.random.rand(84*84*4), np.random.rand(84*84*4), reward, 3, done, 0.95)
#  print(dqn.get_conv1_dw().shape)
# print(o.shape)
# for i in range(4):
#     f = o[0, i]
#     print(f2b(f))

