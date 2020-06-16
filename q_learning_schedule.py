import numpy as np
import random
from collections import defaultdict, OrderedDict, namedtuple
from Environment import Environment
import itertools
import matplotlib.pyplot as plt
Transition = namedtuple('Transition',['s','a','r','s_next','done'])

class QLearning(object):
    def __init__(self, config):
        self.Q_value = defaultdict(float)
        self.env = Environment(config=config)
        self.config = config
        self.reward_averaged = []
        self.reward_history = []
        self.best_result = 0
        self.best_schedule = None

    def act(self, state):
        """
        Pick the best action according to Q values = argmax_a Q(s,a)
        Exploration is forced by epsilon-greedy
        """
        self.state_info, actions, max_value = self.env.generatePossibleAction(state)
        # initial state info is {(0, 'CVD'): (), (1, 'CVD'): (), (2, 'CVD'): (), (3, 'CVD'): (), (4, 'CVD'): (), (0, 'CDO'): (), (1, 'CDO'): (), (2, 'CDO'): ()}
        # import pdb; pdb.set_trace()
        # epsilon greedy function is to capture certain chance of random choices of action
        if self.eps > 0. and np.random.rand() < self.eps:
            # select the action randomly
            return random.choice(actions)
        # import pdb; pdb.set_trace()
        qvals = {action: self.Q_value[self.state_info, action] for action in actions}

        max_q = max(qvals.values())

        # in case of multiple actions having the same Q values
        actions_with_max_q = [a for a,q in qvals.items() if q == max_q]
        # print("Current max action : ",actions_with_max_q)
        return random.choice(actions_with_max_q)

    def _update_q_values(self, tr):
        # since the action space is changing every time we must adhere to that too
        # now the new state or tr.s_next has possible actions which should be retrieved
        # print("Current state", tr.s)
        # print("future_state", tr.s_next)
        self.new_state_info,actions,value = self.env.generatePossibleAction(tr.s_next)
        # import pdb; pdb.set_trace()
        if actions == []:
            actions = [(-1,-1)]
        # Argmax Q value
        max_q_next = max([self.Q_value[self.new_state_info, a] for a in actions])

        # we do not include the values of the next state if terminated
        self.Q_value[self.state_info, tr.a] += self.config['ALPHA'] * (tr.r + self.config['GAMMA'] * max_q_next * (1-tr.done) - self.Q_value[self.state_info, tr.a])
        print("Q value table :",len(self.Q_value))
        return

    def train(self):
        self.eps = self.config['EPSILON']
        print("Inside training data")
        for episode in range(self.config['N_EPISODES']):
            self.obs = self.env.reset()
            # import pdb; pdb.set_trace()
            step = 0
            done = False
            reward_collected = self.config['INIT_REWARD']
            while not done:
                action_to_take = self.act(self.obs) # Action should be only releases lot according to obs
                self.new_obs, number_cqt_lot, done, info = self.env.step(action_to_take)
                reward = -number_cqt_lot*self.config['CQT_penalty']
                # Edit reward to throughput*self.config['INIT_throughput_reward'] - cqt*self.config['INIT_cqt_penalty']
                if done:
                    lot_completed = self.env.completedLotNumber()
                    print("Miss CQT lot ",self.env.miss_cqt)
                    print("Completed : ",lot_completed)
                    if lot_completed == self.config['NUM_JOBS']:
                        print("Completed all lot")
                        reward_collected += 200
                    print("Reward value",reward_collected)
                self._update_q_values(Transition(self.obs, action_to_take, reward, self.new_obs, done))
                self.obs = self.new_obs
                step+=1
                reward_collected += reward
                # print("Collected  ",reward_collected)
            # Decreasing epsilon value for training . Started with total random
            self.eps = self.eps - 2/self.config['N_EPISODES'] if self.eps > 0.01 else 0.01
            print("Epsilon Value :",self.eps)
            if reward_collected > self.best_result:
                self.best_result = reward_collected
                self.best_schedule = self.config['SCHEDULE_ACTION']
            self.reward_history.append(reward_collected)
            self.reward_averaged.append(np.average(self.reward_history[-10:]))
        print("Best result : ",self.best_result)
        print("Best schedule : ",self.best_schedule)
        reward_matrix = np.zeros(self.config['N_EPISODES'])
        for r in range(self.config['N_EPISODES']):
            reward_matrix[r] = self.reward_history[r]
        plt.plot(reward_matrix)
        plt.show()
            # with open('logfile_25_100.txt','a') as fp:
            #     fp.write("Episode {}\nReward {}\nThroughPut {}\n".format(str(episode), str(reward_collected), str(thrput)))
        print("Reward History :",self.reward_history)
        print("Training Over")
        # print(self.env.config['RELEASE_ACTION'])
        return

if __name__ == '__main__':
    config = {
    'NUM_MACHINES_CVD' : 9, # Initial value to start with for CVD entity
	'NUM_MACHINES_CDO' : 4, # Initial Value to start with for CDO entity
	'NUM_JOBS' : 186, # This is running on historical ASOM lot per day
	'SCHEDULE_ACTION' : [],
	'RELEASE_ACTION' : [],
	'SIMULATION_TIME' : 2880, # One day episode is 1440
	'ALPHA' : 0.1,
	'EPSILON' : 1, # Randomly try other action for exploration
	'N_EPISODES' : 2000,
	'GAMMA' : 0.9,
	'INIT_REWARD' : 100, # Each successful lot out from staging will not minus the initial reward
	'DONE_REWARD' : 100, # If lot miss drumbeat
    'CQT_penalty' : 4 # Randomnly assigned weight for CQT misses
    }
    order_cvd = []
    order_cdo = []

    # Below is for static historical one day data training to setup environment
    # Lot will based on arrival time at staging to register
    for i in range(config['NUM_JOBS']):
        mac_cvd = [i for i in range(config['NUM_MACHINES_CVD'])]
        mac_cvd = random.sample(mac_cvd, config['NUM_MACHINES_CVD'])
        order_cvd.append(tuple(mac_cvd))

    for i in range(config['NUM_JOBS']):
        mac_cdo = [i for i in range(config['NUM_MACHINES_CDO'])]
        mac_cdo = random.sample(mac_cdo, config['NUM_MACHINES_CDO'])
        order_cvd.append(tuple(mac_cdo))

    config['ORDER_OF_PROCESSING_CVD'] = order_cvd
    config['ORDER_OF_PROCESSING_CDO'] = order_cdo

    model = QLearning(config=config)
    model.train()
    print(config['SCHEDULE_ACTION'])
    print(config['RELEASE_ACTION'])
