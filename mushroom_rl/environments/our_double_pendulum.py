import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces


def normalize_angle(x):
    return x % (2 * np.pi)


def ode(x, u):
    # https://underactuated.mit.edu/multibody.html#section1

    g = 9.81
    l1, l2 = 1.0, 1.0
    m1, m2 = 2.0, 2.0
    k1, k2 = 1e-3, 1e-3

    th1, th2, dth1, dth2 = x
    u1, u2 = u

    s1, c1 = np.sin(th1), np.cos(th1)
    s2, c2 = np.sin(th2), np.cos(th2)
    s12 = np.sin(th1 + th2)

    # inertia
    M = np.array(
        [
            [
                (m1 + m2) * l1**2 + m2 * l2**2 + 2.0 * m2 * l1 * l2 * c2,
                m2 * l2**2 + m2 * l1 * l2 * c2,
            ],
            [m2 * l2**2 + m2 * l1 * l2 * c2, m2 * l2**2],
        ]
    )

    # Corliolis
    C = np.array(
        [
            [0.0, -m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2],
            [
                0.5 * m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2,
                -0.5 * m2 * l1 * l2 * dth1 * s2,
            ],
        ]
    )

    # gravity
    tau = -g * np.array([(m1 + m2) * l1 * s1 + m2 * l2 * s12, m2 * l2 * s12])

    B = np.eye(2)

    u1 = u1 - k1 * dth1
    u2 = u2 - k2 * dth2

    u = np.hstack([u1, u2])
    v = np.hstack([dth1, dth2])

    inv_M = np.linalg.inv(M)
    a = inv_M @ (tau + B @ u - C @ v)
    # a = np.linalg.solve(M, tau + B @ u - C @ v)

    return np.hstack((v, a))


def reward_fn(state, action):
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    Q = np.diag(np.array([1e1, 1e1, 1e-1, 1e-1]))
    R = np.diag(np.array([1e-3, 1e-3]))

    q, p, qd, pd = state
    _state = np.hstack((normalize_angle(q), normalize_angle(p), qd, pd))

    cost = (_state - goal).T @ Q @ (_state - goal)
    cost += action.T @ R @ action
    return -0.5 * cost


def observe_fn(state):
    sin_q, cos_q = np.sin(state[0]), np.cos(state[0])
    sin_p, cos_p = np.sin(state[1]), np.cos(state[1])
    return np.hstack([sin_q, cos_q, sin_p, cos_p, state[2], state[3]])


class OurDoublePendulum(Environment):
    def __init__(self, max_action=5.0, dt=0.05, horizon=100, gamma=1.0, seed=1):
        self._state = None
        self.max_action = max_action
        self.np_generator = np.random.default_rng(seed)
        self.state_dim = 4

        observation_space = spaces.Box(-np.inf, np.inf, shape=(6,))
        action_space = spaces.Box(-max_action, max_action, shape=(2,))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)
        super().__init__(mdp_info)

    def reset(self, state=None):
        self._state = np.zeros((self.state_dim,))
        return observe_fn(self._state), {}

    def step(self, action):
        _action = self._bound(action, -self.max_action, self.max_action)
        _action *= 5.0

        _deriv = ode(self._state, _action)
        self._state += _deriv * self.info.dt
        self._state += self.np_generator.normal(0, 1e-2, size=self.state_dim)

        reward = reward_fn(self._state, _action)
        obs = observe_fn(self._state)
        return obs, reward, False, {}
