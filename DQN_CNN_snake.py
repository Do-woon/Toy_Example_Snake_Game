from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from collections import deque
import numpy as np
import random
import copy
import pylab
from snake import App

EPISODES = 200000


class DQNAgent:
    def __init__(self, action_size, state_size):
        self.load_model = False
        self.test_mode = True

        self.state_size = state_size
        self.action_size = action_size

        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.epsilon = self.epsilon_start
        self.exploration_steps = 1000000
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99

        self.memory = deque(maxlen=400000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        if self.load_model or self.test_mode:
            self.model.load_weights("./dqn_snake_model.h5")
        if self.load_model:
            self.epsilon = (self.epsilon_start + self.epsilon_end)/2
        elif self.test_mode:
            self.epsilon = 0.001

        self.update_target_model()

        self.model.compile(optimizer=Adam(lr=1e-3), loss='mse', )
        self.sum_loss = 0

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (4, 4), activation='relu', padding='same', strides=(2, 2), input_shape=self.state_size))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = state.reshape((1,self.state_size[0], self.state_size[1], self.state_size[2]))
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, is_run):
        self.memory.append((state, action, reward, next_state, is_run))

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_states = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        actions, rewards, is_runs = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            is_runs.append(mini_batch[i][4])

        next_q_value = self.target_model.predict(next_states)
        update_target = self.model.predict(states)
        for i in range(self.batch_size):
            if not is_runs[i]:
                update_target[i][actions[i]] = rewards[i]
            else:
                update_target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(next_q_value[i])

        # self.sum_loss += loss[0]
        self.model.fit(states, update_target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


def pre_processing(body_x, body_y, body_len, a_x, a_y, width, height):
    pos_x = [elem // 44 for elem in body_x]
    pos_y = [elem // 44 for elem in body_y]
    pos_x_a = a_x // 44
    pos_y_a = a_y // 44

    s = np.zeros((height + 2, width + 2, 3), dtype='float32')
    # encode apple state
    s[pos_y_a + 1, pos_x_a + 1, 0] = 1.0
    # encode head state
    s[pos_y[0] + 1, pos_x[0] + 1, 1] = 1.0
    # encode body state
    for idx in range(1, len(pos_x)):
        s[pos_y[idx] + 1, pos_x[idx] + 1, 2] = 1.0

    return s


if __name__ == "__main__":
    env = App(verbose=False)
    env.on_init()

    s_size = (env.height + 2, env.width + 2, 3)
    act_size = 4
    agent = DQNAgent(act_size, s_size)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        step, score = 0, 0

        player_x, player_y, _, length, apple_x, apple_y, _ = env.reset()
        state = pre_processing(player_x, player_y, length, apple_x, apple_y, env.width, env.height)

        while env.is_running():
            step += 1
            global_step += 1

            action = agent.get_action(state)

            player_x, player_y, _, length, apple_x, apple_y, reward = env.do_action(action, do_render=agent.test_mode)
            next_state = pre_processing(player_x, player_y, length, apple_x, apple_y, env.width, env.height)

            if not env.is_running():
                reward -= 100
            agent.append_sample(state, action, reward, next_state, env.is_running())
            if len(agent.memory) >= agent.train_start and not agent.test_mode:
                agent.train_model()

            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward
            state = copy.deepcopy(next_state)

        print("episode : ", e, " score : %.1f"%(score), " memory len : ", len(agent.memory),
              " epsilon : ", agent.epsilon, " global step : ", global_step, " end step : ", step,
              " avg loss : ", agent.sum_loss/float(step))
        agent.sum_loss = 0
        scores.append(score)
        episodes.append(e)

        if e % 1000 == 0 and e != 0 and not agent.test_mode:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig('./dqn_score_history.png')
            agent.model.save_weights("./dqn_snake_model.h5")
