from __future__ import print_function
from __future__ import division


class Config:
    def __init__(self):
        # gym's environment name
        self.env_name = 'MontezumaRevengeDeterministic-v3'
        # 'MontezumaRevengeDeterministic-v3' 'PongDeterministic-v3'

        # action size for given environment
        self.action_size = 18   # 18 (MR) | 6 (Pong)

        # size of the input observation (image to pass through 2D Convolution)
        self.state_size = [84, 84, 3]   # Box(210, 160, 3) - default

        # number of threads
        self.threads_num = 2

        # local loop size for one episode
        self.LOCAL_T_MAX = 10   # 10 (MR) | 20 (Pong)

        # learning rate
        self.entropy_beta = 1e-2

        # learning rate
        self.learning_rate = 2e-4

        # maximum global time step
        self.MAX_TIME_STEP = 10 * 10 ** 7

        self.RMSP_ALPHA = 0.99   # decay parameter for RMSProp
        self.RMSP_EPSILON = 0.1  # epsilon parameter for RMSProp

        self.wGAMMA = 0.99   # worker's discount factor for rewards
        self.mGAMMA = 0.999  # manager's discount factor for rewards

        # feudal representation
        self.d = 256    # internal representation size
        self.k = 16     # output representation size
        self.h = 10     # number of manager's cores (horizon)
        self.c = 10     # goal horizon to sum up

cfg = Config()
