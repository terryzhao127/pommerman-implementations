from pommerman.agents import BaseAgent
from utils.general import get_logger, Progbar
from utils.replay_buffer import ReplayBuffer
from collections import deque

import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt


# TODO: Refine codes which are copied.

class DQNAgent(BaseAgent):
    non_terminal_reward = 0

    def __init__(self, env, config, exp_schedule, lr_schedule, is_training_agent, train_from_scratch=False,
                 logger=None):
        """
        Initialize Q Network and env

        :param env: Game environment
        :param config: config(hyper-parameters) instance
        :param logger: logger instance from logging module
        :param exp_schedule: exploration strategy for epsilon
        :param lr_schedule: schedule for learning rate
        """
        super(DQNAgent, self).__init__()

        # Variables initialized in __init__
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._done_mask = None
        self._learning_rate = None
        self._q_values = None
        self._target_q_values = None
        self._next_q_values = None
        self._update_target_op = None
        self._loss = None
        self._train_op = None
        self._grad_norm = None

        # Variables initialized in init_agent
        self._session = None
        self._avg_reward_placeholder = None
        self._max_reward_placeholder = None
        self._std_reward_placeholder = None
        self._avg_q_placeholder = None
        self._max_q_placeholder = None
        self._std_q_placeholder = None
        # TODO: Commented due to lack of evaluate()
        # self._eval_reward_placeholder = None
        self._merged = None
        self._file_writer = None
        self._saver = None
        self._train_replay_buffer = None
        self._train_rewards = None
        self._train_max_q_values = None
        self._train_q_values = None
        self._avg_reward = None
        self._max_reward = None
        self._std_reward = None
        self._avg_q = None
        self._max_q = None
        self._std_q = None
        # TODO: Commented due to lack of evaluate()
        # self._eval_reward = None
        self._time_step = None
        self._progress_bar = None
        self._has_episode_started = None

        # Variables initialized in act.
        self._last_action = None
        self._last_idx = None

        # Directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self._logger = logger
        if logger is None:
            self._logger = get_logger(config.log_path)

        self._config = config
        self._env = env
        self._exp_schedule = exp_schedule
        self._lr_schedule = lr_schedule
        self._is_training_agent = is_training_agent
        self._train_from_scratch = train_from_scratch

        # Build model.
        self._build()

    def init_agent(self, id_, game_type):
        super(DQNAgent, self).init_agent(id_, game_type)

        # Assume the graph has been constructed.
        # Create a tf Session and run initializer of variables.
        self._session = tf.Session()

        # Tensorboard
        self._add_summary()

        # Initialize all variables.
        init = tf.global_variables_initializer()
        self._session.run(init)

        # Synchronise q and target_q networks.
        self._session.run(self._update_target_op)

        # for saving networks weights
        self._saver = tf.train.Saver()

        # Initialize replay buffer and variables.
        self._train_replay_buffer = ReplayBuffer(self._config.buffer_size, self._config.state_history)
        self._train_rewards = deque(maxlen=self._config.num_episodes_test)
        self._train_max_q_values = deque(maxlen=1000)
        self._train_q_values = deque(maxlen=1000)
        self._init_averages()

        self._time_step = 0
        self._progress_bar = Progbar(target=self._config.nsteps_train)

        self._has_episode_started = False

        if not self._train_from_scratch:
            self._load()

    def act(self, obs, action_space):
        state = obs['board'][:, :, None]

        if not self._is_training_agent:
            # Act greedily when testing.
            if self._has_episode_started:
                self._train_replay_buffer.store_effect(
                    self._last_idx,
                    self._last_action,
                    0,
                    done=False
                )

            self._last_idx = self._train_replay_buffer.store_frame(state)
            q_input = self._train_replay_buffer.encode_recent_observation()
            action = self._get_best_action(q_input)[0]
            self._last_action = action

            return action

        if self._has_episode_started:
            self._train(DQNAgent.non_terminal_reward, done=False)

        self._time_step += 1

        # Replay buffer
        idx = self._train_replay_buffer.store_frame(state)
        q_input = self._train_replay_buffer.encode_recent_observation()

        # Choose action according to current Q and exploration
        best_action, self._train_q_values = self._get_best_action(q_input)
        action = self._exp_schedule.get_action(best_action)

        self._train_max_q_values.append(max(self._train_q_values))
        self._train_q_values += list(self._train_q_values)

        self._last_action = action
        self._last_idx = idx

        if not self._has_episode_started:
            self._has_episode_started = True

        return action

    def episode_end(self, reward):
        """
        Updates to perform at the end of an episode
        """
        # Reset episode.
        self._has_episode_started = False

        if not self._is_training_agent:
            return

        self._train(reward, done=True)
        self._train_rewards.append(reward)

        # TODO: Commented due to lack of evaluate() and record()
        # if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
        #     # evaluate our policy
        #     last_eval = 0
        #     print("")
        #     scores_eval += [self.evaluate()]
        #
        # if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
        #     self.logger.info("Recording...")
        #     last_record = 0
        #     self.record()

    def shutdown(self):
        """
        Save trained results
        """
        if not self._is_training_agent:
            return

        self._logger.info("- Training done.")
        self._save()

        # TODO: Commented due to lack of evaluate()
        # scores_eval += [self.evaluate()]
        # DQNAgent.export_plot(scores_eval, "Scores", self.config.plot_output)

    def _train(self, reward, done):
        # Store the transition.
        self._train_replay_buffer.store_effect(
            self._last_idx,
            self._last_action,
            reward,
            done=done
        )

        # Perform a training step.
        loss_eval, grad_eval = self._train_step(
            self._time_step,
            self._train_replay_buffer,
            self._lr_schedule.epsilon
        )

        # Logging
        if self._time_step > self._config.learning_start \
                and self._time_step % self._config.log_freq == 0 \
                and self._time_step % self._config.learning_freq == 0:

            self._update_averages(self._train_rewards, self._train_max_q_values, self._train_q_values)
            self._exp_schedule.update(self._time_step)
            self._lr_schedule.update(self._time_step)
            if len(self._train_rewards) > 0:
                self._progress_bar.update(
                    self._time_step + 1,
                    exact=[
                        ("Loss", loss_eval), ("Avg R", self._avg_reward),
                        ("Max R", np.max(self._train_rewards)),
                        ("eps", self._exp_schedule.epsilon),
                        ("Grads", grad_eval), ("Max Q", self._max_q),
                        ("lr", self._lr_schedule.epsilon)
                    ]
                )

        elif self._time_step < self._config.learning_start and self._time_step % self._config.log_freq == 0:
            sys.stdout.write("\rPopulating the memory {}/{}...".format(self._time_step, self._config.learning_start))
            sys.stdout.flush()

    def _build(self):
        """
        Build model by adding all necessary variables.
        """
        # Add placeholders.
        self._add_placeholders_op()

        # Compute Q values of state.
        states = self._process_state(self._states)
        self._q_values = self._get_q_values_op(states, scope='q', reuse=False)

        # Compute Q values of next state.
        next_states = self._process_state(self._next_states)
        self._target_q_values = self._get_q_values_op(next_states, scope='target_q', reuse=False)

        # for Double DQN
        self._next_q_values = self._get_q_values_op(next_states, scope='q', reuse=True)

        # Add update operator for target network.
        self._add_update_target_op('q', 'target_q')

        # Add square loss.
        self._add_loss_op(self._q_values, self._target_q_values, self._next_q_values)

        # Add optimizer for the main networks.
        self._add_optimizer_op('q')

    def _add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        state_shape = list(self._env.observation_space.shape)

        self._states = tf.placeholder(tf.uint8, (None, 11, 11, self._config.state_history))
        self._actions = tf.placeholder(tf.int32, (None,))
        self._rewards = tf.placeholder(tf.float32, (None,))
        self._next_states = tf.placeholder(tf.uint8, (None, 11, 11, self._config.state_history))
        self._done_mask = tf.placeholder(tf.bool, (None,))
        self._learning_rate = tf.placeholder(tf.float32, ())

    def _process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        :param state:
                Node of tf graph of shape = (batch_size, height, width, nchannels) of type tf.uint8.if,
                values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self._config.high

        return state

    def _get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        :param state: (tf tensor) shape = (batch_size, img height, img width, nchannels)
        :param scope: (string) scope name, that specifies if target network or not
        :param reuse: (bool) reuse of variables in the scope
        :return out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self._env.action_space.n
        out = state

        with tf.variable_scope(scope, reuse=reuse) as _:
            x = layers.conv2d(state, 32, 5, stride=2, padding='SAME')
            x = layers.conv2d(x, 64, 4, stride=2, padding='SAME')
            x = layers.conv2d(x, 64, 3, stride=1, padding='SAME')
            x = layers.flatten(x)
            x = layers.fully_connected(x, 512)
            out = layers.fully_connected(x, num_actions, activation_fn=None)

        return out

    def _add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will
        assign all variables in the target network scope with the values of
        the corresponding variables of the regular network scope.

        :param q_scope: (string) name of the scope of variables for q
        :param target_q_scope: (string) name of the scope of variables
                for the target network
        """
        tar_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        self._update_target_op = tf.group(*[tf.assign(tar_vars[i], q_vars[i]) for i in range(len(tar_vars))])

    def _add_loss_op(self, q, target_q, next_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        :param q: (tf tensor) shape = (batch_size, num_actions)(Q(s, a))
        :param target_q: (tf tensor) shape = (batch_size, num_actions)(Q_target(s', a'))
        :param next_q: Q(s', a') for Double DQN
        """
        num_actions = self._env.action_space.n
        not_done = 1 - tf.cast(self._done_mask, tf.float32)

        # Double DQN
        # need q_next(Q(s', a')), then find argmax in it
        max_a = tf.argmax(next_q, axis=1)
        q_max = tf.reduce_sum(target_q * tf.one_hot(max_a, num_actions), axis=1)
        q_samp = self._rewards + not_done * self._config.gamma * q_max

        # nature DQN
        q_s = tf.reduce_sum(q * tf.one_hot(self._actions, num_actions), axis=1)
        self._loss = tf.reduce_mean(tf.square(q_samp - q_s))

    def _add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads_and_vars = optimizer.compute_gradients(self._loss, vars)

        clip_grads_and_vars = None
        if self._config.grad_clip:
            clip_grads_and_vars = [(tf.clip_by_norm(gv[0], self._config.clip_val), gv[1]) for gv in grads_and_vars]
        self._train_op = optimizer.apply_gradients(clip_grads_and_vars)
        self._grad_norm = tf.global_norm(clip_grads_and_vars)

    def _add_summary(self):
        """
        Tensorflow stuff
        """
        # extra placeholders to log stuff from python
        self._avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self._max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self._std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self._avg_q_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self._max_q_placeholder = tf.placeholder(tf.float32, shape=(), name="max_q")
        self._std_q_placeholder = tf.placeholder(tf.float32, shape=(), name="std_q")

        # TODO: Commented due to lack of evaluate()
        # self._eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self._loss)
        tf.summary.scalar("grads norm", self._grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self._avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self._max_reward_placeholder)
        tf.summary.scalar("Std Reward", self._std_reward_placeholder)

        tf.summary.scalar("Avg Q", self._avg_q_placeholder)
        tf.summary.scalar("Max Q", self._max_q_placeholder)
        tf.summary.scalar("Std Q", self._std_q_placeholder)

        # TODO: Commented due to lack of evaluate()
        # tf.summary.scalar("Eval Reward", self._eval_reward_placeholder)

        # logging
        self._merged = tf.summary.merge_all()
        self._file_writer = tf.summary.FileWriter(self._config.output_path,
                                                  self._session.graph)

    def _init_averages(self):
        """
        Define extra attributes for tensorboard.
        """
        self._avg_reward = -21.
        self._max_reward = -21.
        self._std_reward = 0

        self._avg_q = 0
        self._max_q = 0
        self._std_q = 0

        # TODO: Commented due to lack of evaluate()
        # self._eval_reward = -21.

    def _get_action(self, obs):
        """
        Returns action with some epsilon strategy

        :param obs: observation from gym
        """
        if np.random.random() < self._config.soft_epsilon:
            return self._env.action_space.sample()
        else:
            return self._get_best_action(obs)[0]

    def _get_best_action(self, obs):
        """
        Return best action

        :param obs: 4 consecutive observations from gym
        :return action: (int)
        :return action_values: (np array) q values for all actions
        """
        action_values = self._session.run(self._q_values, feed_dict={self._states: [obs]})[0]
        return np.argmax(action_values), action_values

    def _train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        :param t: (int) nth step
        :param replay_buffer: buffer for sampling
        :param lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # Perform training step
        if t > self._config.learning_start and t % self._config.learning_freq == 0:
            loss_eval, grad_eval = self._update_step(t, replay_buffer, lr)

        # Occasionally update target network with q network
        if t % self._config.target_update_freq == 0:
            self._update_target_params()

        # Occasionally save the weights
        if t % self._config.saving_freq == 0:
            self._save()

        return loss_eval, grad_eval

    def _update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        :param t: number of iteration (episode and move)
        :param replay_buffer: ReplayBuffer instance .sample() gives batches
        :param lr: (float) learning rate
        :return loss: (Q - Q_target) ^ 2
        """
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(self._config.batch_size)

        fd = {
            # Inputs
            self._states: s_batch,
            self._actions: a_batch,
            self._rewards: r_batch,
            self._next_states: sp_batch,
            self._done_mask: done_mask_batch,
            self._learning_rate: lr,

            # Extra info
            self._avg_reward_placeholder: self._avg_reward,
            self._max_reward_placeholder: self._max_reward,
            self._std_reward_placeholder: self._std_reward,
            self._avg_q_placeholder: self._avg_q,
            self._max_q_placeholder: self._max_q,
            self._std_q_placeholder: self._std_q,

            # TODO: Commented due to lack of evaluate()
            # self._eval_reward_placeholder: self.eval_reward,
        }

        loss_eval, grad_norm_eval, summary, _ = self._session.run(
            [self._loss, self._grad_norm, self._merged, self._train_op],
            feed_dict=fd
        )

        # Tensorboard
        self._file_writer.add_summary(summary, t)

        return loss_eval, grad_norm_eval

    def _update_target_params(self):
        """
        Update parameters of Q with parameters of Q
        """
        self._session.run(self._update_target_op)

    def _load(self):
        """
        Loads session
        """
        ckpt = tf.train.get_checkpoint_state(self._config.model_output)
        self._saver.restore(self._session, ckpt.model_checkpoint_path)

    def _save(self):
        """
        Saves session
        """
        if not os.path.exists(self._config.model_output):
            os.makedirs(self._config.model_output)

        model_path = os.path.join(self._config.model_output, 'model.ckpt')
        self._saver.save(self._session, model_path)

    def _update_averages(self, rewards, max_q_values, q_values, scores_eval=None):
        """
        Update the averages

        :param rewards: deque
        :param max_q_values: deque
        :param q_values: deque
        :param scores_eval: list
        """
        self._avg_reward = np.mean(rewards)
        self._max_reward = np.max(rewards)
        self._std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self._max_q = np.mean(max_q_values)
        self._avg_q = np.mean(q_values)
        self._std_q = np.sqrt(np.var(q_values) / len(q_values))

        # TODO: Commented due to lack of evaluate()
        # if len(scores_eval) > 0:
        #     self.eval_reward = scores_eval[-1]

    @staticmethod
    def export_plot(y, y_label, filename):
        """
        Export a plot in filename

        :param y: (list) of float / int to plot
        :param filename: (string) directory
        """
        plt.figure()
        plt.plot(range(len(y)), y)
        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.savefig(filename)
        plt.close()
