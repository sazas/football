import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from gfootball.env import player_base


class Player(player_base.PlayerBase):
  """An agent loaded from PPO2 cnn model checkpoint."""

  def __init__(self, model_path):
    player_base.PlayerBase.__init__(self)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)
    imported_graph = tf.train.import_meta_graph(model_path + '.meta')
    imported_graph.restore(self._sess, model_path)

  def __del__(self):
    self._sess.close()

  def take_action(self, observation):
    observation = _convert_observation(observation, True, True)
    action = self._sess.run("player_0/ppo2_model/ArgMax:0", feed_dict={"player_0/ppo2_model/Ob:0": observation})
    return [int(action[0])]

def _convert_observation(observation, fixed_positions, add_relative):
  def do_flatten(obj):
    """Run flatten on either python list or numpy array."""
    if type(obj) == list:
      return np.array(obj).flatten()
    return obj.flatten()

  final_obs = []
  for obs in observation:
    o = []
    if fixed_positions:
      for i, name in enumerate(['left_team', 'left_team_direction',
                                'right_team', 'right_team_direction']):
        o.extend(do_flatten(obs[name]))
        # If there were less than 11vs11 players we backfill missing values
        # with -1.
        if len(o) < (i + 1) * 22:
          o.extend([-1] * ((i + 1) * 22 - len(o)))
    else:
      o.extend(do_flatten(obs['left_team']))
      o.extend(do_flatten(obs['left_team_direction']))
      o.extend(do_flatten(obs['right_team']))
      o.extend(do_flatten(obs['right_team_direction']))

    # If there were less than 11vs11 players we backfill missing values with
    # -1.
    # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)
    if len(o) < 88:
      o.extend([-1] * (88 - len(o)))

    active_position = np.array(obs['left_team'][obs['active']])
    if add_relative:
      # +44 vals of relative positions
      for i, name in enumerate(['left_team', 'right_team']):
        o.extend(do_flatten(np.array(obs[name]) - active_position))
        # If there were less than 11vs11 players we backfill missing values
        # with -1.
        if len(o) < 88 + (i + 1) * 22:
          o.extend([-1] * (88 + (i + 1) * 22 - len(o)))

    # ball position
    o.extend(obs['ball'])
    if add_relative:
      # +2 relative
      o.extend(np.array(obs['ball'])[0:2] - active_position)
    # ball direction
    o.extend(obs['ball_direction'])
    # one hot encoding of which team owns the ball
    if obs['ball_owned_team'] == -1:
      o.extend([1, 0, 0])
    if obs['ball_owned_team'] == 0:
      o.extend([0, 1, 0])
    if obs['ball_owned_team'] == 1:
      o.extend([0, 0, 1])

    active = [0] * 11
    if obs['active'] != -1:
      active[obs['active']] = 1
    o.extend(active)

    game_mode = [0] * 7
    game_mode[obs['game_mode']] = 1
    o.extend(game_mode)
    final_obs.append(o)
  return np.array(final_obs, dtype=np.float32)
