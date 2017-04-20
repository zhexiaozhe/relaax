class Config:
    def __init__(self):
        # action size for given environment
        self.action_size = 18

        # size of the input observation (image to pass through 2D Convolution)
        self.state_size = [84, 84, 3]

        # number of consecutive observations to stack in state
        self.history_len = 4

        # local loop size for one episode
        self.episode_len = 5

cfg = Config()
