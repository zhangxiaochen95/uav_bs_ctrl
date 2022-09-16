REGISTRY = {}

from algos.madrqn.agents.rnn_agents import RnnAgent
from algos.madrqn.agents.gnn_agents import GnnAgent

REGISTRY['rnn'] = RnnAgent
REGISTRY['gnn'] = GnnAgent