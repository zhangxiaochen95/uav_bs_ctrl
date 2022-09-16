REGISTRY = {}

from algos.drqn.agents.rnn_agents import RnnAgent
from algos.drqn.agents.gnn_agents import GnnAgent

REGISTRY['rnn'] = RnnAgent
REGISTRY['gnn'] = GnnAgent