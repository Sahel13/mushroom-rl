import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces


def normalize_angle(x):
    return x % (2 * np.pi)


def ode(x, u):
    # https://underactuated.mit.edu/acrobot.html#cart_pole

    g = 9.81  # gravity
    l = 0.5  # pole length
    mc = 10.0  # cart mass
    mp = 1.0  # pole mass

    x, q, xd, qd = x

    sth = np.sin(q)
    cth = np.cos(q)

    xdd = (u + mp * sth * (l * qd**2 + g * cth)) / (mc + mp * sth**2)

    qdd = (-u * cth - mp * l * qd**2 * cth * sth - (mc + mp) * g * sth) / (
        l * mc + l * mp * sth**2
    )

    return np.hstack((xd, qd, xdd, qdd))


def reward_fn(state, action):
    goal = np.array([0.0, np.pi, 0.0, 0.0])
    Q = np.diag(np.array([1e0, 1e1, 1e-1, 1e-1]))
    R = 1e-3

    cost = (state - goal).T @ Q @ (state - goal)
    cost += action**2 * R
    return -0.5 * cost


class PsocCartpole(Environment):
    def __init__(
        self, max_action=50.0, dt=0.05, horizon=100, discount_factor=1.0, seed=1
    ):
        self._state = None
        self.max_action = max_action
        self.np_generator = np.random.default_rng(seed)
        self.state_dim = 4

        # MDP properties
        observation_space = spaces.Box(-np.inf, np.inf, shape=(self.state_dim,))
        action_space = spaces.Box(-max_action, max_action, shape=(1,))
        mdp_info = MDPInfo(
            observation_space, action_space, discount_factor, horizon, dt
        )
        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.zeros(self.state_dim)
        else:
            self._state = state
            self._state[1] = normalize_angle(self._state[1])

        return self._state, {}

    def step(self, action):
        u = self._bound(action[0], -self.max_action, self.max_action)
        state_deriv = ode(self._state, u.item())
        self._state += state_deriv * self.info.dt
        self._state += self.np_generator.normal(0, 1e-2, size=self.state_dim)
        self._state[1] = normalize_angle(self._state[1])

        reward = reward_fn(self._state, u.item())
        return self._state, reward, False, {}
