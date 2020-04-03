import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.backend import print_tensor as pt
from tensorflow import keras

_action_noise = 1e-3

class ProbabilityDistribution(tf.keras.Model):
    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)
        mu, sigma = tf.split(x, 2, axis=-1)
        return tf.random.normal(tf.shape(mu), mu, sigma + _action_noise)
        

class ActorCriticModel(keras.Model):
    def __init__(self, action_space, observation_space, learning_rate=3e-4):
        super(ActorCriticModel, self).__init__('mlp_policy')

        self._init_layers(action_space, observation_space)

    def _init_layers(self, action_space, observation_space):

        n_hidden1 = 128
        self.num_actions = action_space.shape[0]

        # init state value network
        self.value_dense = keras.layers.Dense(n_hidden1, activation='relu')
        self.value = keras.layers.Dense(self.num_actions, name='value')

        # init policy network
        self.policy_dense = keras.layers.Dense(n_hidden1, activation='relu')
        self.mu = keras.layers.Dense(1, activation='tanh')
        self.sigma = keras.layers.Dense(1, activation='relu')
        self.dist = ProbabilityDistribution()

    def call(self, inputs, **kwargs):

        x = tf.convert_to_tensor(inputs)
        #x = pt(x, 'inputs')

        # init critic model
        hidden_vals = self.value_dense(x)
        #hidden_vals = pt(hidden_vals, 'hidden')

        # init actor model
        hidden_policy = self.policy_dense(x)
        mu = self.mu(hidden_policy)
        sigma = self.sigma(hidden_policy)
        stats = keras.layers.concatenate([mu, sigma], axis=-1)
        #stats = pt(stats, 'stats')

        return stats, self.value(hidden_vals)

    def action_value(self, obs):
        stats, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(stats)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class Agent(object):
    def __init__(self, model, lr=7e-3, gamma=0.99):
        self.model = model
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=[self._actor_loss, self._critic_loss]
        )

        self.gamma = gamma

    def train(self, env, batch_sz=64, updates=500):

        filename = 'train.csv'
        open(filename, 'w').close()

        with open(filename, 'a') as f:

            # storage
            actions = np.empty((batch_sz,), dtype=np.float32)
            rewards, dones, values = np.empty((3, batch_sz))
            observations = np.empty((batch_sz,) + env.observation_space.shape)

            # training loop
            # 1) collect samples
            # 2) send to optimizer
            # 3) repeat updates times
            ep_rewards = [0.0]
            next_obs = env.reset()
            render = True
            for update in range(updates):
                for step in range(batch_sz):
                    observations[step] = next_obs.copy()
                    actions[step], values[step] = self.model.action_value(next_obs[None, :])
                    next_obs, rewards[step], dones[step], _ = env.step([actions[step]])
                    #print(step, observations[step], actions[step], values[step], rewards[step], dones[step])

                    ep_rewards[-1] += rewards[step]

                    if render and False:
                        env.render()

                    if dones[step]:
                        render = not render
                        ep_rewards.append(0.0)
                        next_obs = env.reset()
                        print("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

                #print('predict next value')
                _, next_value = self.model.action_value(next_obs[None, :])

                td_targets, td_errors = self._targets_errors(rewards, values, dones, next_value)

                #actions = pt(actions, 'actions')
                #td_errors = pt(td_errors, 'td_errors')
                acts_and_errors = keras.layers.concatenate([actions[None, :], td_errors[None, :]], axis=-1, dtype=tf.dtypes.float64)

                losses = self.model.train_on_batch(observations, [acts_and_errors, td_targets])
                print("[%d/%d] Losses: %s" % (update + 1, updates, losses))

                pd.DataFrame(losses).to_csv(f, index=False, header=False)

        return ep_rewards

    def test(self, env, render=True):
        obs, done, ep_reward, step = env.reset(), False, 0, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs_before = obs
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            print(step, ep_reward, obs_before, action[0], reward, done)
            if render:
                env.render()
            step += 1
            
        return ep_reward

    def _targets_errors(self, rewards, values, dones, next_value):
        values = np.append(values, next_value, axis=-1)
        targets = np.zeros_like(rewards)
        errors = np.zeros_like(rewards)

        for t in range(len(rewards)):
            targets[t] = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t])
            errors[t] = targets[t] - values[t]

        #print(targets, errors)
        return targets, errors

    def _critic_loss(self, td_error, value):
        """
        V(St) = value (state-value network output) of state at time t
        yt (TD target) = r + gamma * V(St+1)
        delta (TD error) = (yt - V(St))^2
        delta = (r + gamma * V(St+1) - V(St))^2
        delta = (y_pred - y_true)^2 (MSE?)
        """
        return keras.losses.mean_squared_error(td_error, value)

    def _actor_loss(self, acts_and_errors, stats):
        """
        Pi(St) = action (policy network output) given state at time t
        loss = -log(Pi(St)) * delta
        loss = -log(N(a|mu(St), sig(St))) * delta
        """
        acts, errors = tf.split(acts_and_errors, 2, axis=-1)
        #acts = pt(acts, 'acts')
        #errors = pt(errors, 'errors')

        #stats = pt(stats, 'stats')
        mu, sigma = tf.split(stats, 2, axis=-1)
        #mu = pt(mu, 'mu')
        #sigma = pt(sigma, 'sigma')

        norm = tfp.distributions.Normal(mu, sigma + _action_noise)
        #norm = pt(norm, 'norm')

        log_prob = norm.log_prob(acts)
        #log_prob = pt(log_prob, 'log_probs')

        loss = -1 * log_prob * errors
        #loss = pt(loss, 'actor loss')

        return loss

def main():
    env = gym.make('MountainCarContinuous-v0')
    model = ActorCriticModel(env.action_space, env.observation_space)
    agent = Agent(model)

    rewards_history = agent.train(env)
    print("Finished training, testing...")
    print("Total reward: %03d" % agent.test(env, render=True))

if __name__ == '__main__':
  main()
