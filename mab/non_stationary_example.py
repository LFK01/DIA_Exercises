import matplotlib.pyplot as plt
from mab.Non_stationary_environment import *
from mab.SWTS_Learner import *
from mab.TS_Learner import *

# set number of arms
n_arms = 4
# set probabilities for each phase for each arm
p = np.array([[0.15, 0.1, 0.2, 0.35],
              [0.35, 0.21, 0.2, 0.35],
              [0.5, 0.1, 0.1, 0.15],
              [0.8, 0.21, 0.1, 0.15]])

# set time horizon
T = 400

# n experiments set to 100 since we want to compute an average over all the results
n_experiments = 100
# list to collect experiments from Thompson Sampling Learner
ts_rewards_per_experiment = []
# list to collect experiments from Sliding Window Thompson Sampling Learner
swts_rewards_per_experiment = []
window_size = int(np.sqrt(T))

# iterate over the number of experiments
for e in range(0, n_experiments):
    # create an object for the environment of the Thompson Sampling Learner
    ts_env = Non_Stationary_Environment(n_arms, probabilities=p, horizon=T)
    # create the TS_Learner
    ts_learner = TS_Learner(n_arms=n_arms)

    # create an object for the environment of the Sliding Window Thompson Sampling
    swts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    # create the Sliding Windows TS Learner
    swts_learner = SWTS_Learner(n_arms=n_arms, window_size=window_size)

    # iterate over all the rounds
    for t in range(0, T):
        # decide what arm to pull in the Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        print('TS t=' + str(t) + ' and pulled_arm=' + str(pulled_arm))
        # receives the reward from the TS environment based on the pulled arm
        reward = ts_env.round(pulled_arm=pulled_arm)
        # updates the learner
        ts_learner.update(pulled_arm, reward)

        # decide what arm to pull in the Sliding Window Thompson Sampling Learner
        pulled_arm = swts_learner.pull_arm()
        print('SW-TS t=' + str(t) + ' and pulled_arm=' + str(pulled_arm))
        # receives the reward from the SW-TS environment based on the pulled arm
        reward = swts_env.round(pulled_arm)
        # updates the learner
        swts_learner.update(pulled_arm, reward)

    # saves the rewards
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    swts_rewards_per_experiment.append(swts_learner.collected_rewards)

# setup vectors to save the regret of the two algorithms
ts_instantaneous_regret = np.zeros(T)
swts_instantaneous_regret = np.zeros(T)

n_phases = len(p)  # set number of phases taken from the probabilities matrix
# compute the length of a phase
phases_len = int(T / n_phases)

# compute the optimal value for each phase
opt_per_phases = p.max(axis=1)
# create a matrix where to store the optimal value for each round
optimum_per_round = np.zeros(T)

avg_regrets_ts = np.mean(ts_rewards_per_experiment, axis=0)
avg_regrets_swts = np.mean(swts_rewards_per_experiment, axis=0)

# iterate over all the phases, set the optimum per round for each phases
for i in range(0, n_phases):
    optimum_per_round[i * n_phases: (i + 1) * phases_len] = opt_per_phases[i]
    # compute instantaneous regret for Thompson Sampling for each round
    # by computing the difference between the optimum per phases and the minimum of the reward collected by the
    # Thompson Sampling Learner
    ts_instantaneous_regret[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i] - np.mean(
                                                                    ts_rewards_per_experiment, axis=0)[
                                                                        i * phases_len: (i + 1) * phases_len]
    swts_instantaneous_regret[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i] - np.mean(
                                                                    ts_rewards_per_experiment, axis=0)[
                                                                        i * phases_len: (i + 1) * phases_len]

# plots the graphs of the rewards
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(swts_rewards_per_experiment, axis=0), 'b')
plt.plot(optimum_per_round, '--k')
plt.legend(["TS", "SW-TS", "Optimum"])
plt.show()

# plots the graphs of the regrets
plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(ts_instantaneous_regret, axis=0), 'r')
plt.plot(np.cumsum(swts_instantaneous_regret, axis=0), 'b')
plt.legend(["TS", "SW-TS"])
plt.show()