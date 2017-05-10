import deepmind_lab
import numpy as np
import random

from scipy.misc import imresize


class Lab(object):
    def __init__(self, level, fps=20, width=84, height=84, frame_skip=4,
                 display=False, action_size='m', no_op_max=0):
        self._frame_skip = frame_skip
        self._no_op_max = no_op_max
        self._width = width
        self._height = height
        self._display = display
        self._actions = ACTIONS[action_size].values()
        if display:
            width = 640
            height = 480

        self.env = deepmind_lab.Lab(
            level, ['RGB_INTERLACED'],
            config={
                'fps': str(fps),
                'width': str(width),
                'height': str(height)
            })

        self.s_t = None
        self.reset()

    def state(self):
        return self.s_t

    def act(self, action):
        reward, terminal, self.s_t = self._process_frame(self._actions[action])
        return reward, terminal

    def _process_frame(self, action):
        reward = self.env.step(action, num_steps=self._frame_skip)
        terminal = not self.env.is_running()

        if terminal:
            return reward, terminal, None

        # train screen shape is (84, 84, 3) by default
        x_t = self.env.observations()['RGB_INTERLACED']
        if self._display:
            x_t = imresize(x_t, (self._width, self._height))

        x_t = x_t.astype(np.float32)
        x_t *= (1.0 / 255.0)
        return reward, terminal, x_t

    def reset(self):
        while True:
            self.env.reset()

            # randomize initial state
            if self._no_op_max > 0:
                no_op = np.random.randint(0, self._no_op_max + 1)
                for _ in range(no_op):
                    action = random.choice(self._actions)
                    self.env.step(action, num_steps=self._frame_skip)

            _, terminal, self.s_t = self._process_frame(random.choice(self._actions))
            if not terminal:
                break


def _action(*entries):
    return np.array(entries, dtype=np.intc)

FULL_ACTIONS = {
        'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
        'look_right': _action(20, 0, 0, 0, 0, 0, 0),
        'look_up': _action(0, 10, 0, 0, 0, 0, 0),
        'look_down': _action(0, -10, 0, 0, 0, 0, 0),
        'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
        'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
        'forward': _action(0, 0, 0, 1, 0, 0, 0),
        'backward': _action(0, 0, 0, -1, 0, 0, 0),
        'fire': _action(0, 0, 0, 0, 1, 0, 0),
        'jump': _action(0, 0, 0, 0, 0, 1, 0),
        'crouch': _action(0, 0, 0, 0, 0, 0, 1)
}

SMALL_ACTIONS = {
        'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
        'look_right': _action(20, 0, 0, 0, 0, 0, 0),
        'forward': _action(0, 0, 0, 1, 0, 0, 0)
}

MEDIUM_ACTIONS = {
        'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
        'look_right': _action(20, 0, 0, 0, 0, 0, 0),
        'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
        'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
        'forward': _action(0, 0, 0, 1, 0, 0, 0),
        'backward': _action(0, 0, 0, -1, 0, 0, 0)
}

ACTIONS = {'f': FULL_ACTIONS,
           'full': FULL_ACTIONS,
           'b': FULL_ACTIONS,
           'big': FULL_ACTIONS,
           'm': MEDIUM_ACTIONS,
           'medium': MEDIUM_ACTIONS,
           's': SMALL_ACTIONS,
           'small': SMALL_ACTIONS}
