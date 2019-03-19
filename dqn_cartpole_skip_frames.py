import gym
import tensorflow as tf
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.deepq.replay_buffer import ReplayBuffer
import random
from typing import List, Tuple

tf.reset_default_graph()

# Utility functions
def backup_training_variables(scope: str, sess: tf.Session) -> List[List]:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return sess.run(variables)


def restore_training_variables(
    scope: str, vars_values: List[List], sess: tf.Session
) -> None:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    ops = []
    for i, var in enumerate(variables):
        ops.append(tf.assign(var, vars_values[i]))
    sess.run(tf.group(*ops))


def render_env(state, env):
    env.s = state
    env.render()


# functions for graduatlly decrease learning rate using linear functions
def get_explore_rate(t):
    return max(
        (EPSILON_MIN - EPSILON_MAX) * t / MAX_EXPLORATION_RATE_DECAY_TIMESTEP
        + EPSILON_MAX,
        EPSILON_MIN,
    )


# Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
ALPHA = 1e-3  # learning rate
EPSILON_MAX = 1  # exploration rate
EPSILON_MIN = 0.05
GAMMA = 0.99  # discount factor
MAX_EXPLORATION_RATE_DECAY_TIMESTEP = 40000
TARGET_NETWORK_UPDATE_STEP_FREQUENCY = 500
EXPERIENCER_REPLAY_BATCH_SIZE = 32

# Training parameters
SEED = 1000
NUM_EPISODES = 1000000
MAX_NUM_STEPS = 200
TOTAL_MAX_TIMESTEPS = 400000
# we picked 2000 because on average, the random agent would make a successful drop-off after 2848.14
# timesteps according to https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

## Initialize env
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
env = gym.make(
    "CartPole-v0"
).env  # without the .env, there is gonna be a 200 max num steps.
random.seed(SEED)
env.seed(SEED)
np.random.seed(SEED)
tf.random.set_random_seed(SEED)


def build_neural_network(scope: str) -> Tuple[tf.Variable]:
    with tf.variable_scope(scope):
        # Because this is discrete(500) observation space, we actually need to use the one-hot
        # tensor to make training easier.
        # https://github.com/hill-a/stable-baselines/blob/a6f7459a301a7ba3c4bbcebff5829ea054ae802f/stable_baselines/common/input.py#L20
        # So, instead of
        # observation = tf.placeholder(tf.float32, [None, 1], name="observation")
        # We use
        observation = tf.placeholder(shape=(None,) + env.observation_space.shape, dtype=tf.float32)
        pred = tf.placeholder(tf.float32, [None], name="pred")
        q_value_index = tf.placeholder(tf.int32, [None], name="q_value_index")
        fc1 = tf.contrib.layers.fully_connected(
            inputs=observation,
            num_outputs=64,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
        fc2 = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=64,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
        fc3 = tf.contrib.layers.fully_connected(
            inputs=fc2,
            num_outputs=env.action_space.n,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
        max_q_value = tf.math.reduce_max(fc3, axis=1)
        # https://github.com/hill-a/stable-baselines/blob/88a5c5d50a7f6ad1f44f6ef0feaa0647ed2f7298/stable_baselines/deepq/build_graph.py#L394
        q_value = tf.reduce_sum(
            fc3 * tf.one_hot(q_value_index, env.action_space.n), axis=1
        )
        loss = tf.losses.mean_squared_error(q_value, pred)

        # Manual gradient desent step, which is equivalent to
        train_opt = tf.train.GradientDescentOptimizer(ALPHA).minimize(loss)
        grads = tf.gradients(loss, tf.trainable_variables(scope))
        vars_and_grads = list(zip(tf.trainable_variables(scope), grads))
        ops = []
        for item in vars_and_grads:
            ops.append(tf.assign(item[0], item[0] - ALPHA * item[1]))
        train_opt = tf.group(*ops)

        #
        saver = tf.train.Saver()
        tf.summary.scalar("Loss", loss)
        write_op = tf.summary.merge_all()
    return (
        fc3,
        observation,
        pred,
        max_q_value,
        q_value,
        loss,
        train_opt,
        saver,
        write_op,
        q_value_index,
    )


(
    fc3,
    observation,
    pred,
    max_q_value,
    q_value,
    loss,
    train_opt,
    saver,
    write_op,
    q_value_index,
) = build_neural_network("q_network")
(
    target_fc3,
    target_observation,
    target_pred,
    target_max_q_value,
    target_q_value,
    target_loss,
    target_train_opt,
    target_saver,
    target_write_op,
    target_q_value_index,
) = build_neural_network("target_network")

# Start the training process
sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./logs", sess.graph_def)
restore_training_variables(
    "target_network", backup_training_variables("q_network", sess), sess
)
random_actions_taken = 0
er = ReplayBuffer(50000)
episode_rewards = []
finished_episodes_count = 0
target_network_update_counter = 0
total_timesteps = 0
for i_episode in range(NUM_EPISODES):
    raw_state = env.reset()
    done = False
    episode_reward = 0
    for t in range(MAX_NUM_STEPS):
        epsilon = get_explore_rate(total_timesteps)
        total_timesteps += 1
        target_network_update_counter += 1
        # env.render()
        if random.random() < epsilon:
            action = random.randint(0, env.action_space.n - 1)
            random_actions_taken += 1
        else:
            evaluated_fc3 = sess.run(fc3, feed_dict={observation: [raw_state]})
            action = np.argmax(evaluated_fc3[0])
        old_raw_state = raw_state
        raw_state, reward, done, info = env.step(action)
        episode_reward += reward

        # Store transition in the experience replay
        if done == 1:
            done_int = 1
        else:
            done_int = 0
        er.add(old_raw_state, action, reward, raw_state, done_int)

        # Sample random minibatch of trasitions from the experience replay
        if total_timesteps < 1000:
            continue
        obses_t, actions, rewards, obses_tp1, dones = er.sample(32)

        if done:
            finished_episodes_count += 1

        # Predict
        evaluated_target_max_q_value = sess.run(
            target_max_q_value, feed_dict={target_observation: obses_tp1}
        )
        y = rewards + GAMMA * evaluated_target_max_q_value * (1 - dones)

        # Train
        _, summary = sess.run(
            [train_opt, write_op],
            feed_dict={observation: obses_t, pred: y, q_value_index: actions},
        )  # the 0-index column is the old_raw_state

        writer.add_summary(summary, total_timesteps)

        # Update the target network
        if target_network_update_counter > TARGET_NETWORK_UPDATE_STEP_FREQUENCY:
            restore_training_variables(
                "target_network", backup_training_variables("q_network", sess), sess
            )
            target_network_update_counter = 0

        if done:
            break

        if total_timesteps % 10000 == 0:
            save_path = saver.save(sess, "./tmp/model.ckpt")
            print("Model saved in path: %s" % save_path)

    #            if total_timesteps > 15000:
    #                print("debug")

    print(
        "Episode: ",
        i_episode,
        "finished with rewards of ",
        episode_reward,
        "with successful drop-offs of",
        finished_episodes_count,
        "random_actions_taken",
        random_actions_taken,
        "epsilon",
        epsilon,
    )
    episode_rewards += [episode_reward]
    if total_timesteps > TOTAL_MAX_TIMESTEPS:
        break

plt.plot(episode_rewards)
