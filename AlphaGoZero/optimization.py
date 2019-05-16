from __future__ import annotations
from pommerman import constants

import tensorflow as tf
import settings
import os
import numpy as np


class PolicyValueNet:
    default_model_path = './models/'

    def __init__(self, mini_batch_size, trained_model=None) -> None:
        # TODO: Change the number of feature maps
        self._num_feature_maps = 6
        self._l2_penalty_beta = 1e-4
        self._possible_actions = [a for a in constants.Action]

        # Build the network
        self._build()

        if trained_model is not None:
            self.load_model()

    def optimize(self, data):
        # TODO: Process states
        state_batch, mcts_prs, winner_batch = data
        # winner_batch = []
        # state_batch = []
        # mcts_prs = []
        learning_rate = None

        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self._sess.run(
            [self._loss, self._entropy, self._optimizer],
            feed_dict={self._input_states: state_batch,
                       self._mcts_prs: mcts_prs,
                       self._value: winner_batch,
                       self._learning_rate: 0.01}
        )
        return loss, entropy

    def predict(self, state):
        # TODO: Process state
        log_action_prs, value = self._sess.run(
            [self._action_fc, self._evaluation_fc_2],
            feed_dict={self._input_states: state}
        )
        action_prs = list(zip(self._possible_actions, np.exp(log_action_prs[0])))

        return action_prs, value

    def _build(self):
        # TODO: Rewrite this section with Keras API
        """Build the neural network"""
        self._input_states = tf.placeholder(
            tf.float32,
            shape=(None, self._num_feature_maps, settings.board_size, settings.board_size)
        )

        self._input_state = tf.transpose(self._input_states, [0, 2, 3, 1])

        # CNN layers
        self._conv_1 = tf.layers.conv2d(
            inputs=self._input_state,
            filters=32,
            kernel_size=(3, 3),
            padding='same', data_format='channels_last',
            activation=tf.nn.relu
        )
        self._conv_2 = tf.layers.conv2d(
            inputs=self._conv_1,
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu
        )
        self._conv_3 = tf.layers.conv2d(
            inputs=self._conv_2,
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu)

        # Policy head
        self._action_conv = tf.layers.conv2d(
            inputs=self._conv_3,
            filters=4,
            kernel_size=(1, 1),
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu
        )
        self._action_conv_flat = tf.reshape(
            self._action_conv, (-1, 4 * settings.board_size * settings.board_size)
        )
        self._action_fc = tf.layers.dense(
            inputs=self._action_conv_flat,
            units=6,
            activation=tf.nn.log_softmax
        )

        # Value head
        self._evaluation_conv = tf.layers.conv2d(
            inputs=self._conv_3,
            filters=2,
            kernel_size=(1, 1),
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu
        )
        self._evaluation_conv_flat = tf.reshape(
            self._evaluation_conv, (-1, 2 * settings.board_size * settings.board_size)
        )
        self._evaluation_fc_1 = tf.layers.dense(
            inputs=self._evaluation_conv_flat,
            units=64,
            activation=tf.nn.relu
        )
        self._evaluation_fc_2 = tf.layers.dense(
            inputs=self._evaluation_fc_1,
            units=1,
            activation=tf.nn.tanh
        )

        # Value loss
        self._value = tf.placeholder(tf.float32, shape=(None, 1))
        self._value_loss = tf.losses.mean_squared_error(
            self._value,
            self._evaluation_fc_2
        )

        # Policy Loss
        self._mcts_prs = tf.placeholder(
            tf.float32, shape=(None, 6)
        )
        self._policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(
            tf.multiply(self._mcts_prs, self._action_fc), 1
        )))

        # L2 penalty
        variables = tf.trainable_variables()
        l2_penalty = self._l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name.lower()]
        )

        # The loss function
        self._loss = self._value_loss + self._policy_loss + l2_penalty

        # Optimizer
        self._learning_rate = tf.placeholder(tf.float32)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss)

        # Make a session
        self._sess = tf.Session()

        # Calc policy entropy, only for monitoring
        self._entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(
            tf.exp(self._action_fc) * self._action_fc, 1
        )))

        # Initialization
        init = tf.global_variables_initializer()
        self._sess.run(init)

        # For saving model
        self._saver = tf.train.Saver()

    def load_model(self, path=default_model_path):
        self._saver.restore(self._sess, path)

    def save_model(self):
        if not os.path.exists(PolicyValueNet.default_model_path):
            os.makedirs(PolicyValueNet.default_model_path)
        self._saver.save(self._sess, PolicyValueNet.default_model_path)


def _process_data():
    """Process generated self-play data"""
    pass
