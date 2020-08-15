import random
import copy
import numpy as np
from collections import deque
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
import pylab
from snake import App

EPISODES = 20000


class DQNAgent:
    def __init__(self, action_size, state_size):
        self.load_model = False
        self.test_mode = True

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = 1e-3
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.epsilon = self.epsilon_start
        self.exploration_steps = 50000
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
            self.epsilon = self.epsilon_end
            self.train_start = 1000
        elif self.test_mode:
            self.epsilon = 0
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='normal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='normal'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, s):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            s = s.reshape((1, self.state_size))
            q_value = self.model.predict(s)
            return np.argmax(q_value[0])

    def append_sample(self, s, a, r, next_s, is_run):
        self.memory.append((s, a, r, next_s, is_run))

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, is_runs = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            is_runs.append(mini_batch[i][4])

        update_target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            if not is_runs[i]:
                update_target[i][actions[i]] = rewards[i]
            else:
                update_target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, update_target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


# state definition
# 1. (x, y) coordinate of head : 2 dim
# 2. current direction of head(one-hot) : 4 dim
# 2. (dx, dy) of head -> apple : 2 dim
# 3. (dx, dy) of head -> (body lies on each directions) : 2x4 dim
# total : 16 dimensions
# above values will be normalized
def pre_processing(body_x, body_y, head_dir, body_len, a_x, a_y, width, height, action_size):
    pos_x = [elem // 44 for elem in body_x]
    pos_y = [elem // 44 for elem in body_y]
    pos_x_a = a_x // 44
    pos_y_a = a_y // 44

    head_x = pos_x[0]
    head_y = pos_y[0]

    dir_candidate = list(range(0, action_size))

    body_proximity = []
    for d in dir_candidate:
        temp = []
        if d == 0:  # right
            for x in range(head_x+1, width+1):
                if (x, head_y) in zip(pos_x, pos_y):
                    temp.append((x, head_y))
                    break
            if not temp:
                temp.append((width, head_y))
        elif d == 1:  # left
            for x in range(width, head_x, -1):
                if (x, head_y) in zip(pos_x, pos_y):
                    temp.append((x, head_y))
                    break
            if not temp:
                temp.append((-1, head_y))
        elif d == 2:  # up
            for y in range(head_y+1, height+1):
                if (head_x, y) in zip(pos_x, pos_y):
                    temp.append((head_x, y))
                    break
            if not temp:
                temp.append((head_x, height))
        elif d == 3:  # down
            for y in range(height, head_y, -1):
                if (head_x, y) in zip(pos_x, pos_y):
                    temp.append((head_x, y))
                    break
            if not temp:
                temp.append((head_x, -1))

        body_proximity.append(temp)

    s = np.asarray([
        head_x/width, head_y/height,  # 1. (x, y) coordinate of head : 2 dim

        # 2. current direction of head (one_hot) : 4 dim
        float(head_dir == 0), float(head_dir == 1), float(head_dir == 2), float(head_dir == 3),

        (pos_x_a - head_x)/width, (pos_y_a - head_y)/height,  # 2. (dx, dy) of head -> apple : 2 dim

        # 3. (dx, dy) of head -> (body lies on 3 directions) : 2x4 dim
        (body_proximity[0][0][0] - head_x) / width, (body_proximity[0][0][1] - head_y) / height,
        (body_proximity[1][0][0] - head_x) / width, (body_proximity[1][0][1] - head_y) / height,
        (body_proximity[2][0][0] - head_x) / width, (body_proximity[2][0][1] - head_y) / height,
        (body_proximity[3][0][0] - head_x) / width, (body_proximity[3][0][1] - head_y) / height
    ])

    return s


if __name__ == "__main__":
    env = App(verbose=False)
    env.on_init()

    s_size = 16
    act_size = 4
    agent = DQNAgent(act_size, s_size)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        step, score = 0, 0

        player_x, player_y, direction, length, apple_x, apple_y, _ = env.reset()
        current_state = pre_processing(player_x, player_y, direction, length, apple_x, apple_y,
                                       env.width, env.height, act_size)

        while env.is_running():
            step += 1
            global_step += 1

            action = agent.get_action(current_state)

            player_x, player_y, direction, length, apple_x, apple_y, reward \
                = env.do_action(action, do_render=agent.test_mode)

            next_state = pre_processing(player_x, player_y, direction, length, apple_x, apple_y,
                                        env.width, env.height, act_size)

            if not env.is_running():
                reward -= 100
            agent.append_sample(current_state, action, reward, next_state, env.is_running())
            if len(agent.memory) >= agent.train_start and not agent.test_mode:
                agent.train_model()

            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward
            state = copy.deepcopy(next_state)

        print("episode : ", e, " score : %.1f"%score, " memory len : ", len(agent.memory),
              " epsilon : %.2f"%agent.epsilon, " global step : ", global_step, " end step : ", step)
        scores.append(score)
        episodes.append(e)

        if e % 1000 == 0 and e != 0 and not agent.test_mode:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig('./dqn_score_history.png')
            agent.model.save_weights("./dqn_snake_model.h5")
