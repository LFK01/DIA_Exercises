from mab.Environment import *
import numpy as np

class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)
        self.t = 0
        self.horizon = horizon

    # takes as input a pulled arm and return a stochastic reward based on the pulled arm and the phase
    def round(self, pulled_arm):
        # number of phases is equal to the rows of the probabilities matrix
        n_phases = len(self.probabilities)
        # phases size defines the length of a phase
        phases_size = self.horizon / n_phases
        # current_phase computes the phase of the current round
        current_phase = int(self.t / phases_size)

        # the probability of the arm is taken from the probability matrix by number of phases and pulled arm
        print('ROUND: pulled_arm=' + str(pulled_arm))
        p = self.probabilities[current_phase][pulled_arm]
        # the current round is updated
        self.t += 1
        # the function returns a binomial distribution of the arm considered the current probability
        return np.random.binomial(1, p)
