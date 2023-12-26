import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .rewards import asymmetric_gaussian


class SpaceEnv(gym.Env):

    def __init__(self, agent_orbits, target_orbits, metric):
        metadata = {'render.modes' : []}
        super(SpaceEnv, self).__init__()


        self.num_agents = 1
        self.num_targets = target_orbits.shape[2]

        self.agent_orbits = agent_orbits
        self.target_orbits = target_orbits

        self.observation_space = spaces.Dict({"current action" : spaces.Discrete(self.num_targets + 1), "num repeat actions": spaces.Discrete(agent_orbits.shape[0]+1)})
        self.action_space = spaces.Discrete(self.num_targets + 1)

        self.metric = metric
        
        self.tstep = 0
        self.agent_state = dict({"current action" : 0, "num repeat actions":0})
        self.action_history = []

    def _get_obs(self):
        return self.agent_state
    
    def get_info(self):
        return {"tstep" : self.tstep, "available_actions" : self.available_actions(), "action_history" : self.action_history}

    def calc_reward(self, action, available_actions):
        if action == 0 and len(available_actions) > 1:
            reward = -1
        elif action == 0:
            reward = 0
        else:
            n_repeated = self._get_nrepeated()
            reward = asymmetric_gaussian(n_repeated)

        return reward
    
    def _get_nrepeated(self):
        n_repeated = 0

        for i in np.arange(2, len(self.action_history)+1):
            if self.action_history[-i] == self.agent_state["current action"]:
                n_repeated+=1
            else:
                break

        return n_repeated

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set the timestep to 0
        self.tstep = 0
        self.agent_state = dict({"current action" : 0, "num repeat actions":0})
        self.action_history = []

        observation = self._get_obs()
        info = self.get_info()

        return observation, info
    

    def step(self, action):

        prev_available_actions = self.available_actions()

        if self.action_space.contains(action):
            self.agent_state["current action"] = action
            self.action_history.append(action)

            self.agent_state["num repeat actions"] = self._get_nrepeated()
        else:
           raise ValueError(f"invalid value of action, must be an integer from 0 to {self.num_targets}")

        self.tstep += 1

        terminated = self.tstep == self.agent_orbits.shape[0]

        reward = self.calc_reward(action, prev_available_actions)

        observation = self._get_obs()
        info = self.get_info()


        return observation, reward, terminated, False, info
    
    def render(self, mode = None):
        return
    
    def close(self):
        return
    
    def action_mask(self):

        mask = np.zeros(shape=self.num_targets+1, dtype=bool)

        if self.tstep < self.agent_orbits.shape[0]:

            r_agent = self.agent_orbits[self.tstep, :]
            r_targets = self.target_orbits[self.tstep, :, :].T


            mask[1:] = self.metric.applyv(r_agent, r_targets)
            mask[0] = True

        return mask

    
    def available_actions(self):
            
        mask = self.action_mask()

        avail_actions = np.arange(start=0, stop=self.num_targets+1)

        avail_actions = avail_actions[mask]

        
        return avail_actions



        