from __future__ import annotations
from self_play import SelfPlay
from optimization import PolicyValueNet
from evaluator import Evaluator
from collections import deque
import random

best_net_filename = 'best.model'
new_net_filename = 'new.model'

# Self Play
self_play_num_simulations = 100
num_pipeline_iterations = 1500
self_play_data_size = 10000

# Optimization
optimization_mini_batch_size = 400
optimization_freq = self_play_data_size // 20

# Evaluation
evaluation_freq = 20
evaluation_num_simulations = 100
evaluation_num_games = 20

# Optimization component
best_net = PolicyValueNet(
    mini_batch_size=optimization_mini_batch_size,
    num_epochs=5,
    learning_rate=2e-3,
    lr_multiplier=1.0,
    kl_targ=0.02
)
new_net = PolicyValueNet(
    mini_batch_size=optimization_mini_batch_size,
    num_epochs=5,
    learning_rate=2e-3,
    lr_multiplier=1.0,
    kl_targ=0.02
)

best_net.save_model(best_net_filename)

# Self-play data buffer
data_buffer = deque(maxlen=self_play_data_size)

for i in range(num_pipeline_iterations):
    print(('-' * 20 + 'Iteration %d begins' + '-' * 20) % (i + 1))

    # Self-play component
    self_play_component = SelfPlay(
        best_net,
        num_games=1,
        num_simulations=self_play_num_simulations,
        num_exploration_steps=15
    )

    # Self-play
    print('====Self-play starts====')

    initial_data_count = 0
    while len(data_buffer) != data_buffer.maxlen:
        # Collect initial data
        new_data = self_play_component.start()
        if len(data_buffer) + len(new_data) > self_play_data_size:
            # Abandon redundant data
            data_buffer.extend(random.sample(new_data, k=self_play_data_size - len(data_buffer)))
        else:
            data_buffer.extend(new_data)
        initial_data_count = len(data_buffer)

        print('[Self Play] Collecting initial self play data: %d / %d' % (initial_data_count, self_play_data_size))

    new_data_count = 0
    while new_data_count < optimization_freq and initial_data_count == 0:
        # Collect new data
        new_data = self_play_component.start()
        if new_data_count + len(new_data) > optimization_freq:
            # Abandon redundant data
            data_buffer.extend(random.sample(new_data, k=optimization_freq - new_data_count))
            new_data_count = optimization_freq
        else:
            data_buffer.extend(new_data)
            new_data_count += len(new_data)
        print('[Self Play] Collecting new self play data: %d / %d' % (new_data_count, optimization_freq))

    # Optimization
    print('====Optimization starts====')
    new_net.optimize(data_buffer)

    if (i + 1) % evaluation_freq == 0:
        new_net.save_model(new_net_filename)

        # Evaluation
        print('====Evaluation starts====')
        evaluator = Evaluator(
            best_net,
            new_net,
            num_games=evaluation_num_games,
            num_simulations=evaluation_num_simulations
        )
        if evaluator.start():
            print('[Evaluation] !!!!Find new best network at iteration %d!!!!' % (i + 1))
            new_net.save_model(best_net_filename)
            best_net.load_model(best_net_filename)
        else:
            print('[Evaluation] No new best net.')
