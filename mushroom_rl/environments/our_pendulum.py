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
    eta = 0.5

    q = normalize_angle(state[0])
    dq = state[1]

    _state = np.hstack((q, dq))
    cost = (_state - goal).T @ Q @ (_state - goal)
    cost += R * action**2 
    return - 0.5 * eta * cost


def observe_fn(state):
    sin_q = np.sin(state[0])
    cos_q = np.cos(state[0])
    dq = state[1]
    return np.hstack((sin_q, cos_q, dq))


class OurPendulum(Environment):
    def __init__(
        self, max_action=5.0, dt=0.05, horizon=100, gamma=1.0, seed=1
    ):
        self._state = None
        self.max_action = max_action
        self.np_generator = np.random.default_rng(seed)
        self.state_dim = 2

        observation_space = spaces.Box(-np.inf, np.inf, shape=(3,))
        action_space = spaces.Box(-max_action, max_action, shape=(1,))

        mdp_info = MDPInfo(
            observation_space, action_space, gamma, horizon, dt
        )
        super().__init__(mdp_info)

    def reset(self, state=None):
        self._state = np.zeros((self.state_dim,))
        return observe_fn(self._state), {}

    def step(self, action):
        _action = self._bound(action[0], -self.max_action, self.max_action)

        _deriv = ode(self._state, _action.item())
        self._state += _deriv * self.info.dt
        self._state += self.np_generator.normal(0, 1e-2, size=self.state_dim)

        reward = reward_fn(self._state, _action.item())
        obs = observe_fn(self._state)
        return obs, reward, False, {}
