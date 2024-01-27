import numpy as np
import gym
import time

from monte_carlo_learning import LOOP_SIZE, state_representation, get_max

# state = (position, velocity)
m = 3  # No. of possible actions
EPISODES = 50


def get_random_policy_for_all_states():
    positions = [state_representation(x) for x in np.arange(-1.2, 0.7, 0.1)]
    velocities = [state_representation(x) for x in np.arange(-0.7, 0.7, 0.1)]
    policy = {}
    for position in positions:
        policy[position] = {}
        for velocity in velocities:
            policy[position][velocity] = [1/m for _ in range(m)]
    
    return policy, (positions, velocities)


def initialized_q(states):
    positions = states[0]
    velocities = states[1]
    q_val = {}
    for position in positions:
        q_val[position] = {}
        for velocity in velocities:
            q_val[position][velocity] = []
            for _ in range(3):
                if position == 0.6:
                    q_val[position][velocity].append(0)
                else:
                    q_val[position][velocity].append(np.random.uniform(0, 1))
    return q_val


def behaviour_policy(state, q_val):
    action_prob = behaviour_policy_prob(state, q_val)
    return np.random.choice(range(m), p=action_prob)


def behaviour_policy_prob(state, q_val, epsilon=1):
    state = (state_representation(state[0]), state_representation(state[1]))
    action_prob = np.ones(m, dtype=float) * epsilon / m
    best_action = get_max(q_val[state[0]][state[1]])
    action_prob[best_action] += (1.0 - epsilon)
    return action_prob


def target_policy_prob(state, q_val, epsilon=.3):
    state = (state_representation(state[0]), state_representation(state[1]))
    action_prob = np.ones(m, dtype=float) * epsilon / m
    best_action = get_max(q_val[state[0]][state[1]])
    action_prob[best_action] += (1.0 - epsilon)
    return action_prob


def simulate_env_td_lambda(env):
    _, states = get_random_policy_for_all_states()
    Q = initialized_q(states)
    alpha = 0.01
    gamma = 0.1
    n = 3
    for index in range(EPISODES):
        start_episode = time.time()
        buffer_s = []
        buffer_a = []
        buffer_r = []
        state = env.reset()
        state = (state_representation(state[0]), state_representation(state[1]))
        action = behaviour_policy(state, Q)
        buffer_s.append(state)
        buffer_a.append(action)

        T = LOOP_SIZE
        t = 0
        while True:
            if t < T:
                state_, reward, done, _ = env.step(action)
                state_ = (state_representation(state_[0]), state_representation(state_[1]))
                buffer_s.append(state_)
                buffer_r.append(reward)

                if state_[0] == 0.6:
                    T = t + 1
                else:
                    action_ = behaviour_policy(state, Q)
                    buffer_a.append(action_)
                    action = action_
            tao = t - n + 1

            if tao >= 0:
                rho = 1
                for i in range(tao+1, min(T, tao+n)):
                    rho *= target_policy_prob(buffer_s[i], Q)[buffer_a[i]] /\
                           behaviour_policy_prob(buffer_s[i], Q)[buffer_a[i]]
                G = 0
                for i in range(tao+1, min(tao+n, T)):
                    G += gamma**(i-tao-1) * buffer_r[i-1]

                if tao+n < T:
                    G += gamma ** n * Q[buffer_s[tao + n][0]][buffer_s[tao + n][1]][buffer_a[tao + n]]

                Q[buffer_s[tao][0]][buffer_s[tao][1]][buffer_a[tao]] += \
                    alpha * rho * (G - Q[buffer_s[tao][0]][buffer_s[tao][1]][buffer_a[tao]])

            if tao == T-1:
                break
            env.render()
            t += 1
        print("End of episode {}. Elapsed time: {}".format(index, time.time()-start_episode))
    print("GG.")
    env.destroy()


if __name__ == '__main__':
    environment = gym.make('MountainCar-v0')
    environment.reset()

    simulate_env_td_lambda(environment)

