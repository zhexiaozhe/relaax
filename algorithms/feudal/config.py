class Config:
    def __init__(self):
        # gym's environment name
        self.env_name = 'PongDeterministic-v3'
        # 'MontezumaRevengeDeterministic-v3' 'PongDeterministic-v3'

        # action size for given environment
        self.action_size = 6   # 18 (MR) | 6 (Pong)

        # size of the input observation (image to pass through 2D Convolution)
        self.state_size = [84, 84, 3]   # Box(210, 160, 3) - default

        # number of threads
        self.threads_num = 8

        # local loop size for one episode
        self.episode_len = 10   # 10 (MR) | 20 (Pong)

        # learning rate
        self.learning_rate = 7e-4


cfg = Config()
