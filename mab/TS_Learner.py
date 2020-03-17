from Learner import *


class TS_Learner(Learner):
    def __init__(self, n_arms):  # this class inherits from Learner so we have to pass the same parameters
        # in the constructor
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pulled_arm(self):
        idx = np.argmax(numpy.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_obeservations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward
