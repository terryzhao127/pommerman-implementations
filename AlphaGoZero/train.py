from __future__ import annotations
from self_play import SelfPlay
from optimization import PolicyValueNet
from evaluator import Evaluator

net = PolicyValueNet(1)
self_play_component = SelfPlay(net, num_games=1, num_simulations=2)
evaluator = Evaluator(net, net, num_games=1, num_simulations=1)

self_play_data = self_play_component.start()
net.optimize(self_play_data)
win_ratios = evaluator.start()
