from mab.TS_Learner import *
import numpy as np


class SWTS_Learner(TS_Learner):
    # windows size is used to consider only recent distibutions in the beta parameters estimations
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size

    def update(self, pulled_arm, reward):
        # update round
        self.t += 1
        # update observations on the currently pulled arm
        self.update_observations(pulled_arm, reward)
        # compute cumulative sum of all the previous rewards
        cum_rew = np.sum(self.rewards_per_arm[pulled_arm][-self.window_size:])
        # compute how many times this arm has been pulled
        n_rounds_arm = len(self.rewards_per_arm[pulled_arm][-self.window_size:])

        # update parameters of the beta distribution
        self.beta_parameters[pulled_arm, 0] = cum_rew + 1.0
        self.beta_parameters[pulled_arm, 1] = n_rounds_arm - cum_rew
