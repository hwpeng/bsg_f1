import gym
import numpy as np
from PIL import Image

def process_frame84(frame):
    '''
    From https://github.com/transedward/pytorch-dqn/blob/1ffda6f3724b3bb37c3195b09b651b1682d4d4fd/utils/atari_wrapper.py#L108
    '''
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    img = Image.fromarray(img)
    resized_screen = img.resize((84, 110), Image.BILINEAR)
    resized_screen = np.array(resized_screen)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84])
    #  return x_t.astype(np.uint8)
    return x_t

class gym_env:
    def __init__(self, name="CartPole-v1"):
        self.env = gym.make(name)
        self.env.seed(2020)
        self.name = name
        self.obs_space = self.env.observation_space.shape[0]
        self.act_space = self.env.action_space.n
        #  print('Making gym RL environment:', name)

    def reset(self):
        state = self.env.reset()
        if ('CartPole' not in self.name):
            state = process_frame84(state)
            state = np.stack([state] * 4, axis=0)
        return state

    def step(self, action=0):
        next_state, reward, done, __ = self.env.step(action)
        if ('CartPole' not in self.name):
            next_state = process_frame84(next_state)
            rd = np.array([reward, done])
            rd = np.append(rd, [0]*82)
            rd = rd.reshape(1,84)
            cat_all = np.concatenate((next_state, rd), axis=0).astype(np.float32)
        else:
            cat_all = np.append(next_state, [reward, done]).astype(np.float32)
        return cat_all 
