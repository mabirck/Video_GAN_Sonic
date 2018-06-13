import random
import torch
import numpy as np
from model import RandomPolicy

class Sample(object):
    def __init__(self, state):
        # TODO: Check what are the effects of this
        #utils.assert_eq(type(state), type(next_state))

        #self._state = (state * 255.0).astype(np.uint8)
        self._state = np.array(state).astype(np.uint8)

    @property
    def state(self):
        return self._state.astype(np.float32) / 255.0

    @property
    def next_state(self):
        return self._next_state.astype(np.float32) / 255.0

    def __repr__(self):
        info = ('S(mean): %3.4f'
                % (self.state.mean()))
        return info


class ReplayMemory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.samples = []
        self.oldest_idx = 0

    def __len__(self):
        return len(self.samples)

    def _evict(self):
        """Simplest FIFO eviction scheme."""
        to_evict = self.oldest_idx
        self.oldest_idx = (self.oldest_idx + 1) % self.max_size
        return to_evict

    def burn_in(self, env, num_steps):
        # TODO CHECK
        policy = RandomPolicy(env) # uniform policy
        i = 0

        env.reset()
        while i < num_steps:
            action = policy.get_action()
            state, reward, done, _ = env.step(action)
            self.append(state)
            print(done)
            if True in done:
                state = env.reset()

            i += 1
            if i % 10000 == 0:
                print('%d frames burned in' % i)
        print('%d frames burned into the memory.' % i)

    def append(self, states):
        # Normalize to images [-1, 1]
        states = normalize_frames(states)

        assert len(self.samples) <= self.max_size
        new_samples = [Sample(state) for state in states]

        for new_sample in new_samples:
            if len(self.samples) == self.max_size:
                avail_slot = self._evict()
                self.samples[avail_slot] = new_sample
            else:
                self.samples.append(new_sample)

    def sample(self, batch_size):
        """Simpliest uniform sampling (w/o replacement) to produce a batch.
        """
        assert batch_size < len(self.samples), 'no enough samples to sample from'
        return random.sample(self.samples, batch_size)

    def clear(self):
        self.samples = []
        self.oldest_idx = 0


def samples_to_tensors(samples):
    num_samples = len(samples)

    states_shape = (num_samples, ) + samples[0].state.shape
    states = np.zeros(states_shape, dtype=np.float32)

    for i, s in enumerate(samples):
        states[i] = s.state

    states = torch.from_numpy(states).cuda()

    return states

def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].
    @param frames: A numpy array. The frames to be converted.
    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames

def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.
    @param frames: A numpy array. The frames to be converted.
    @return: The denormalized frames.
    """
    new_frames = frames + 1
    new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)
