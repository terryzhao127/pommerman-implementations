from __future__ import annotations
from self_play import SelfPlay
from optimization import Net, TrainingNet
from evaluator import Evaluator
from collections import deque
from multiprocessing import Process, Manager
import random

best_net_filename = 'best.model'
new_net_filename = 'new.model'

num_pipeline_iterations = 1500

# Self Play
self_play_num_simulations = 1
self_play_data_size = 10
self_play_exploration_steps = 15
self_play_process_count = 14

# Optimization
optimization_mini_batch_size = 4
optimization_freq = self_play_data_size // 10

# Evaluation
evaluation_freq = 1
evaluation_num_simulations = 100
evaluation_num_games = 1
evaluation_max_process_count = 10


def main():
    # Create a file for new model
    p = Process(target=create_new_best_model)
    p.start()
    p.join()

    # Self-play data buffer
    data_buffer = deque(maxlen=self_play_data_size)

    training_params = None

    # Main pipeline
    for i in range(num_pipeline_iterations):
        print(('-' * 20 + 'Iteration %d begins' + '-' * 20) % (i + 1))

        # Self-play
        print('====Self-play starts====')

        initial_data_count = 0
        while len(data_buffer) != data_buffer.maxlen:
            # Collect initial data
            new_data = generate_self_play_data(
                num_simulations=self_play_num_simulations,
                num_exploration_steps=self_play_exploration_steps,
                process_count=self_play_process_count
            )
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
            new_data = generate_self_play_data(
                num_simulations=self_play_num_simulations,
                num_exploration_steps=self_play_exploration_steps,
                process_count=self_play_process_count
            )
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
        with Manager() as manager:
            op_l = manager.list()
            p = Process(target=optimize, args=(data_buffer, training_params, op_l))
            p.start()
            p.join()

            op_l = [x for x in op_l]
            training_params = op_l[0]

        if (i + 1) % evaluation_freq == 0:
            # Evaluation
            print('====Evaluation starts====')
            if evaluate_net(evaluation_num_games, evaluation_num_simulations):
                print('[Evaluation] !!!!Find new best network at iteration %d!!!!' % (i + 1))
                p = Process(target=save_new_best)
                p.start()
                p.join()
            else:
                print('[Evaluation] No new best net.')


def create_new_best_model():
    best_net = TrainingNet(
        mini_batch_size=optimization_mini_batch_size,
        num_epochs=5,
        learning_rate=2e-3,
        lr_multiplier=1.0,
        kl_targ=0.02
    )
    best_net.save_model(best_net_filename)


def optimize(data, training_params, op_l):
    new_net = TrainingNet(
        mini_batch_size=optimization_mini_batch_size,
        num_epochs=5,
        learning_rate=2e-3,
        lr_multiplier=1.0,
        kl_targ=0.02,
        trained_model=new_net_filename
    )
    if training_params is not None:
        new_net.import_params(training_params)
    new_net.optimize(data)
    new_net.save_model(new_net_filename)
    op_l += [new_net.export_params()]


def save_new_best():
    new_net = Net(new_net_filename)
    new_net.save_model(best_net_filename)


def evaluate_net(num_games, num_simulations):
    def run(l):
        best_net = Net(best_net_filename)
        new_net = Net(new_net_filename)
        evaluate = Evaluator(best_net, new_net, num_games=1, num_simulations=num_simulations)
        result = evaluate.start()
        l.append(result)

    if num_games > evaluation_max_process_count:
        process_count = evaluation_max_process_count
        num_iteration = (num_games // process_count) + 1
        left_games = num_games - process_count * (num_iteration - 1)
    else:
        process_count = num_games
        num_iteration = 1
        left_games = 0

    results = []
    for i in range(num_iteration):
        if num_iteration != 1 and i == num_iteration - 1:
            process_count = left_games

        with Manager() as manager:
            shared_list = manager.list()

            processes = [Process(target=run, args=(shared_list,))
                         for _ in range(process_count)]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            results += [_i for _i in shared_list]
    best_net_win_count = sum([x[0] for x in results])
    new_net_win_count = sum([x[1] for x in results])
    win_count = best_net_win_count + new_net_win_count
    win_ratio = [best_net_win_count / win_count, new_net_win_count / win_count]
    print('[Evaluation] Win ratios:', win_ratio)

    return win_ratio[1] > 0.55


def generate_self_play_data(num_simulations, num_exploration_steps, process_count):
    def run(l_):
        best_net = Net(trained_model=new_net_filename)
        self_play_component = SelfPlay(best_net, num_games=1, num_simulations=num_simulations,
                                       num_exploration_steps=num_exploration_steps)
        _new_data = self_play_component.start()
        l_ += _new_data
        print('[Self Play] Collected data in a process, length:', len(_new_data))

    with Manager() as manager:
        shared_list = manager.list()

        processes = [Process(target=run, args=(shared_list,)) for _ in range(process_count)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return [_i for _i in shared_list]


if __name__ == '__main__':
    main()
