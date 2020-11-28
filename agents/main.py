import os
import sys

#model_name = "11_vs_11_easy_stochastic_v2"
model_name = "simple_ppo2"
KAGGLE_PATH = "/kaggle_simulations/agent"
if os.path.exists(KAGGLE_PATH):
    # On kaggle
    sys.path.insert(1, KAGGLE_PATH)
    path = KAGGLE_PATH + f"/saved_models/{model_name}/{model_name}"
else:
    sys.path.insert(1, "/home/alex/Dropbox/projects/python/kaggle/football/agents/")
    path = f"/home/alex/Dropbox/projects/python/kaggle/football/saved_models/{model_name}/{model_name}"

from ppo2_v3 import Player
# Load previously trained Tensorflow model.
player = Player(path)


def agent(obs):
    global player
    # Get observations for the first (and only one) player we control.
    obs = obs['players_raw']
    
    # Execute our agent to obtain action to take.
    agent_output = player.take_action(obs)
    return agent_output