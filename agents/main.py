import os
import sys

file_name = "11_vs_11_easy_stochastic_v2"
KAGGLE_PATH = "/kaggle_simulations/agent"
if os.path.exists(KAGGLE_PATH):
    # On kaggle
    sys.path.insert(1, KAGGLE_PATH)
    path = KAGGLE_PATH + "/saved_models/" + file_name
else:
    sys.path.insert(1, "/home/alex/Dropbox/projects/python/kaggle/football/agents/")
    path = f"/home/alex/Dropbox/projects/python/kaggle/football/saved_models/{file_name}"

from ppo2_v2 import Player
# Load previously trained Tensorflow model.
player = Player(path)

def agent(obs):
    global player
    # Get observations for the first (and only one) player we control.
    obs = obs['players_raw']
    
    # Execute our agent to obtain action to take.
    agent_output = player.take_action(obs)
    return agent_output