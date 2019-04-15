from pommerman.agents import BaseAgent
from pommerman.constants import Action


class IdleAgent(BaseAgent):
    def act(self, obs, action_space):
        return Action.Stop
