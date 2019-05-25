from __future__ import annotations
from pommerman import constants
from tensorflow import logging

import tensorflow as tf
import settings
import os
import numpy as np
import random

logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Net:
    default_model_path = './models/'

    def __init__(self, trained_model=None) -> None:
        # TODO: Change the number of feature maps
        self._num_feature_maps = 7
        self._l2_penalty_beta = 1e-4
        self._possible_actions = [a for a in constants.Action]

        # Build the network
        self._build()

        if trained_model is not None:
            self.load_model(filename=trained_model)

    def predict(self, states):
        log_action_prs, value = self._sess.run(
            [self._action_fc, self._evaluation_fc_2],
            feed_dict={self._input_states: states}
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
        gpu_options = tf.GPUOptions(allow_growth=True)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Calc policy entropy, only for monitoring
        self._entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(
            tf.exp(self._action_fc) * self._action_fc, 1
        )))

        # Initialization
        init = tf.global_variables_initializer()
        self._sess.run(init)

        # For saving model
        self._saver = tf.train.Saver()

    def load_model(self, filename, path=default_model_path):
        tf.reset_default_graph()
        self._saver.restore(self._sess, path + filename)

    def save_model(self, filename, path=default_model_path):
        if not os.path.exists(path):
            os.makedirs(path)
        self._saver.save(self._sess, path + filename)


class TrainingNet(Net):
    def __init__(self,
                 mini_batch_size,
                 num_epochs,
                 learning_rate,
                 lr_multiplier,
                 kl_targ,
                 trained_model=None) -> None:
        # TODO: Change the number of feature maps
        super(TrainingNet, self).__init__(trained_model=trained_model)

        self._mini_batch_size = mini_batch_size
        self._num_epochs = num_epochs
        self._lr_input = learning_rate
        self._lr_multiplier = lr_multiplier
        self._kl_targ = kl_targ

    def export_params(self):
        return {
            'learning_rate': self._lr_input,
            'lr_multiplier': self._lr_multiplier,
            'kl_targ': self._kl_targ,
        }

    def import_params(self, params):
        self._lr_input = params['learning_rate']
        self._lr_multiplier = params['lr_multiplier']
        self._kl_targ = params['kl_targ']

    def optimize(self, self_play_data):
        for i in range(self._num_epochs):
            kl = 0
            loss = 0
            entropy = 0

            for _ in range(len(self_play_data) // self._mini_batch_size):

                mini_batch = random.sample(self_play_data, self._mini_batch_size)

                state_batch = [data[0] for data in mini_batch]
                mcts_prs_batch = [data[1] for data in mini_batch]
                winner_batch = [data[2] for data in mini_batch]

                old_prs, old_v = self.predict(state_batch)
                processed_old_prs = []
                for prs in old_prs:
                    processed_old_prs.append(prs[1])
                processed_old_prs = np.asarray(processed_old_prs)

                winner_batch = np.reshape(winner_batch, (-1, 1))
                loss, entropy, _ = self._sess.run(
                    [self._loss, self._entropy, self._optimizer],
                    feed_dict={
                        self._input_states: state_batch,
                        self._mcts_prs: mcts_prs_batch,
                        self._value: winner_batch,
                        self._learning_rate: self._lr_input * self._lr_multiplier
                    }
                )

                new_prs, new_v = self.predict(state_batch)

                processed_new_prs = []
                for prs in new_prs:
                    processed_new_prs.append(prs[1])
                processed_new_prs = np.asarray(processed_new_prs)

                kl = np.mean(np.sum(
                    (processed_old_prs * (np.log(processed_old_prs + 1e-10) - np.log(processed_new_prs + 1e-10)))[None],
                    axis=1)
                )
                if kl > self._kl_targ * 4:
                    print('[Optimization] Epoch %d break due to D_KL divergence' % (i + 1))
                    # Stop early if D_KL diverges badly
                    break
            print('[Optimization] Epoch %d -' % (i + 1), 'kl: %f' % kl, 'loss: %f' % loss, 'entropy: %f' % entropy)
            # Adaptively adjust the learning rate
            if kl > self._kl_targ * 2 and self._lr_multiplier > 0.1:
                self._lr_multiplier /= 1.5
            elif kl < self._kl_targ / 2 and self._lr_multiplier < 10:
                self._lr_multiplier *= 1.5
