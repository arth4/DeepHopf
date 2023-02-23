import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

from functools import lru_cache, wraps
from tensorforce.environments import Environment
from os import path


class HopfieldEnvironment(Environment):
    max_actions = 1000
    def __init__(self, size=10, weights=None, symmetric=False):
        self.size = size
        self.weights = weights if weights is not None else np.random.normal(size=(size, size))
        if symmetric:
            self.weights = (self.weights + self.weights.T) / 2
        self.weights = self.normalize(self.weights)
        self.reset()
        super().__init__()

    @classmethod
    def _set_max_actions(cls, max_actions):
        cls.max_actions = max_actions


    def index2coord(self, index):
        return (index % self.size, index // self.size)

    def save(self, fn):
        state_info = np.array(sorted(list(self._state_dict.values()), key=lambda x: x[3]))
        np.savez(fn, weights=self.weights, state=state_info, last_reward=self.last_reward, action_count=self.action_count)
        
    def coord2id(self, coord):
        assert coord in self._state_dict, f"coord {coord} not in state_dict!!!! \n {self._state_dict}"
        return self._state_dict[coord][3]

    @classmethod
    def load(cls, fn):
        data = np.load(fn)
        size = data["weights"].shape[0]
        env = cls(size=size, weights=data["weights"])
        env._state_dict = {(i, j): [i, j, angle, id] for (i, j, angle, id) in data["state"]}
        env.cached_state = None
        env.dirty_implicit_weights = set()
        env.last_reward = data["last_reward"]
        env.action_count = data["action_count"]
        return env

    def swap(self, i, j, angle):
        """Swaps pos of i and j and rotates i by angle"""
        self.cached_state = None
        i_coord = self.index2coord(i) if np.isscalar(i) else tuple(i)
        j_coord = self.index2coord(j) if np.isscalar(j) else tuple(j)

        assert i_coord in self._state_dict, f"i {i} -> {i_coord} not in state_dict!!!! \n {self._state_dict}"
        self.dirty_implicit_weights.add(self.coord2id(i_coord))
        if i_coord != j_coord:
            if j not in self._state_dict:
                self._state_dict[j_coord] = list(j_coord) + self._state_dict[i_coord][2:]
                del self._state_dict[i_coord]
            else:
                self.dirty_implicit_weights.add(self.coord2id(j_coord))
                self._state_dict[i_coord][2:], self._state_dict[j_coord][2:] = self._state_dict[j_coord][2:], self._state_dict[i_coord][2:]
        self._state_dict[j_coord][2] = (self._state_dict[j_coord][2] + angle) % np.pi 

    def reset(self):
        self._state_dict = {(i, i): [i, i, 0, i] for i in range(self.size)}
        self.last_reward = 0
        self.cached_state = None
        self.action_count = 0
        self.cached_implicit_weights = None
        self.dirty_implicit_weights = set()
        return self.state

    @property
    def state(self):
        """Returns a numpy array of the positions and angles sorted by id"""
        if self.cached_state is None:
            self.cached_state = np.array(sorted(list(self._state_dict.values()), key=lambda x: x[3]))[:, :3]

        return self.cached_state

    @property
    def pos(self):
        return self.state[:, :2]

    @property
    def angle(self):
        return self.state[:, 2]

    def states(self):
        return dict(type='float', shape=(self.size, 3))

    def normalize(self, x):
        min, max = np.min(x), np.max(x)
        return (x - min) / (max - min) if max != min else np.zeros_like(x)

    def do_action(self, action):
        start_pos = self.pos[action["index"]]
        angle = np.pi * (2 ** action["angle"]) / 180  
        self.swap(start_pos, (action["newX"], action["newY"]), angle)

    def make_random_action(self):
        index = np.random.randint(self.size)
        newX = np.random.randint(self.size)
        newY = np.random.randint(self.size)
        angle = np.random.randint(8)
        return {"index": index, "newX": newX, "newY": newY, "angle": angle}

    def execute(self, actions):
        self.do_action(actions)
        self.action_count += 1
        terminal = self.action_count >= self.max_actions
        reward = self.reward(terminal)
        diff = reward - self.last_reward
        self.last_reward = reward
        return self.state, terminal, diff

    def actions(self):
        return {"index": dict(type='int', num_values=self.size),
                "newX": dict(type='int', num_values=self.size),
                "newY": dict(type='int', num_values=self.size),
                "angle": dict(type='int', num_values=8)}

    def reward(self, terminal):
        return -self.calculate_loss()

    def calculate_loss(self):
        return np.sum(np.abs(self.weights - self.normalize(self.implicit_weights)))

    @property
    def implicit_weights(self):
        if self.cached_implicit_weights is None:
            self.cached_implicit_weights = self._implict_weights()
        self._fix_dirty_implicit_weights()
        return self.cached_implicit_weights

    def _fix_dirty_implicit_weights(self):
        while self.dirty_implicit_weights:
            ind = self.dirty_implicit_weights.pop()
            for i in range(self.size):
                w_ij = self.implict_weight(i, ind)
                self.cached_implicit_weights[i, ind] = w_ij
                self.cached_implicit_weights[ind, i] = w_ij




    def _implict_weights(self):
        """Returns the weights implied by the current state"""
        weights = np.zeros((self.size, self.size))
        self.m = np.column_stack((np.cos(self.state[:, 2]), np.sin(self.state[:, 2])))
        for i in range(self.size):
            for j in range(i + 1):
                w_ij = self.implict_weight(i, j)
                weights[i, j] = w_ij
                weights[j, i] = w_ij
        return weights

   

    def implict_weight(self, i, j):
        """Returns the weight from j to i implied by the current state"""
        if i == j:
            return 0
        r = self.state[j][:2] - self.state[i][:2]
        m = self.m
        dist = sqrt(r.dot(r)) 
        h_dip_1 = -m[j] / dist**3
        h_dip_2 = 3 * r * m[j].dot(r) / dist**5
        h_dip = h_dip_1 + h_dip_2

        return m[i].dot(h_dip)

    def plot(self):
        plt.quiver(self.pos[:, 0], self.pos[:, 1], np.cos(self.angle), np.sin(self.angle), pivot='mid')


if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    env = HopfieldEnvironment(size=100)
    fig, ax = plt.subplots()

    def animate(i):
        plt.cla()
        plt.title(f"Step {i}")
        env.do_action(env.make_random_action())
        env.plot()
        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        ax.grid(alpha=0.9, linewidth=0.5)

    ani = FuncAnimation(fig, animate, interval=100, frames=100)
    plt.show()
