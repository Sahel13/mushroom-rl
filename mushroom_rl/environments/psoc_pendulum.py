import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces


def normalize_angle(x):
    return x % (2 * np.pi)


def ode(state, action):
    l, m = 1.0, 1.0
    g, d = 9.81, 1e-3

    q, dq = state
    ddq = -g / l * np.sin(q) + (action - d * dq) / (m * l**2)
    return np.hstack((dq, ddq))


def reward_fn(state, action):
    goal = np.array([np.pi, 0.0])
    Q = np.diag(np.array([1e1, 1e-1]))
    R = 1e-3

    cost = (state - goal).T @ Q @ (state - goal)
    cost += action**2 * R
    return -0.5 * cost


class PsocPendulum(Environment):
    def __init__(
        self, max_action=5.0, dt=0.05, horizon=100, discount_factor=1.0, seed=1
    ):
        self._state = None
        self.max_action = max_action
        self.np_generator = np.random.default_rng(seed)
        self.state_dim = 2

        # MDP properties
        observation_space = spaces.Box(-np.inf, np.inf, shape=(self.state_dim,))
        action_space = spaces.Box(-max_action, max_action, shape=(1,))
        mdp_info = MDPInfo(
            observation_space, action_space, discount_factor, horizon, dt
        )
        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            # self._state = np.zeros(self.state_dim)
            rand_angle = self.np_generator.uniform(low=-np.pi / 2, high=np.pi / 2)
            self._state = np.array([rand_angle, 0.0])
        else:
            self._state = state
            self._state[0] = normalize_angle(self._state[0])

        return self._state, {}

    def step(self, action):
        u = self._bound(action[0], -self.max_action, self.max_action)
        state_deriv = ode(self._state, u.item())
        self._state += state_deriv * self.info.dt
        # self._state += self.np_generator.normal(0, 1e-2, size=self.state_dim)
        self._state[0] = normalize_angle(self._state[0])

        reward = reward_fn(self._state, u.item())
        return self._state, reward, False, {}
