import gym
import random

import relaax.client.rlx_client


class GymEnv(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.state = None
        self.reset()

    def act(self, action):
        print('action', repr(action))
        self.state, reward, terminal, _ = self.env.step(action)
        print('state', repr(self.state))
        return reward, terminal

    def reset(self):
        self.env.reset()
        self.state, _, terminal, _ = self.env.step(random.randint(0, 1))
        assert not terminal


if __name__ == "__main__":
    env = GymEnv()

    client = relaax.client.rlx_client.Client('localhost:7001') # connect to RLX server
    action = client.init(env.state)
    while True:
        reward, reset = env.act(action)
        if reset:
            episode_score = client.reset(reward)
            env.reset()
            action = client.send(None, env.state)
        else:
            action = client.send(reward, env.state)
