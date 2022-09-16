from abc import abstractmethod


class MultiAgentEnv(object):
    """Base class for multi-agent environments"""

    def seed(self, seed):
        """Sets random seed for the environment."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the environment to an initial state, and returns initial observations/states."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions):
        """
        Executes one step of interaction between agents and environment.

        Args:
            actions: actions of all agents

        Returns:
            observation: local observations of all agents
            state: global state of the environment
            reward: reward signal of all agents
            done: whether the episode terminates
            info: auxiliary diagnostic information
        """
        raise NotImplementedError

    @abstractmethod
    def get_obs(self):
        """Returns local observations of all agents in a list."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        """Returns a global state characterizing the entire environment."""
        raise NotImplementedError

    def close(self):
        pass

    def render(self):
        raise NotImplementedError

    @ property
    def unwrapped(self):
        return self


class MultiAgentWrapper(object):
    """Base class for multi-agent env wrappers"""

    def __init__(self, env: MultiAgentEnv):
        self.env = env

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def step(self, actions):
        obs, state, rew, done, info = self.env.step(actions)
        obs = self.observation(obs)
        state = self.state(state)
        return obs, state, rew, done, info

    def reset(self):
        obs, state = self.env.reset()
        return self.observation(obs), self.state(state)

    @abstractmethod
    def observation(self, obs):
        raise NotImplementedError

    @abstractmethod
    def state(self, state):
        raise NotImplementedError

    @abstractmethod
    def reward(self, reward):
        raise NotImplementedError

    @property
    def unwrapped(self) -> MultiAgentEnv:
        return self.env.unwrapped
