from __future__ import annotations
from self_play import SelfPlay
from optimization import PolicyValueNet
from evaluator import Evaluator
from collections import deque
from copy import deepcopy

best_net_filename = 'best.model'
new_net_filename = 'new.model'

num_pipeline_iterations = 1500
self_play_data_size = 10000

optimization_freq = self_play_data_size // 20
evaluation_freq = 50

# Optimization component
best_net = PolicyValueNet(
    mini_batch_size=512,
    num_epochs=5,
    learning_rate=2e-3,
    lr_multiplier=1.0,
    kl_targ=0.02
)
new_net = PolicyValueNet(
    mini_batch_size=512,
    num_epochs=5,
    learning_rate=2e-3,
    lr_multiplier=1.0,
    kl_targ=0.02
)

best_net.save_model(best_net_filename)

# Self-play data buffer
self_play_data = deque(maxlen=self_play_data_size)

for i in range(num_pipeline_iterations):
    # Self-play component
    self_play_component = SelfPlay(best_net, num_games=1, num_simulations=50, num_exploration_steps=5)

    # Collect self-play data
    new_data_count = 0
    while new_data_count < optimization_freq:
        while len(self_play_data) != self_play_data.maxlen:
            new_data = self_play_component.start()

            self_play_data.extend(new_data)
            new_data_count += len(new_data)
            print('Data count: %d' % len(self_play_data))

    # Optimization
    new_net.optimize(self_play_data)

    if i + 1 % evaluation_freq == 0:
        new_net.save_model(new_net_filename)

        # Evaluation
        evaluator = Evaluator(best_net, new_net, num_games=10, num_simulations=50)
        if evaluator.start():
            print('========Find new best network at iteration %d========' % (i + 1))
            new_net.save_model(best_net_filename)
            best_net.load_model(best_net_filename)
