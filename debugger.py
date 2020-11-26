# Set up the Environment.
from kaggle_environments import make
env = make("football", configuration={"episodeSteps": 100, "save_video": False,
                                      "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True,
                                      "team_2": 0})
output = env.run(["./agents/main.py", "do_nothing"])
res = output[-1]
print('Left player: reward = %s, status = %s, info = %s' % (res[0]['reward'], res[0]['status'], res[0]['info']))
print('Right player: reward = %s, status = %s, info = %s' % (res[1]['reward'], res[1]['status'], res[1]['info']))
