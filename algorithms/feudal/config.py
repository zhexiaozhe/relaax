class Config:
    def __init__(self):
        # gym's environment name
        self.env_name = 'MontezumaRevengeDeterministic-v3'

        # action size for given environment
        self.action_size = 18

        # size of the input observation (image to pass through 2D Convolution)
        self.state_size = [84, 84, 3]   # Box(210, 160, 3)

        # number of threads
        self.threads_num = 4

        # local loop size for one episode
        self.episode_len = 10


cfg = Config()
