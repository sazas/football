# coding=utf-8


r"""Player from PPO2 mlp checkpoint.

Example usage with play_game script:
python3 -m gfootball.play_game \
    --players "ppo2_mlp:left_players=1,checkpoint=$YOUR_PATH,policy=$POLICY"

$POLICY should be mlp
"""

from baselines.common.policies import build_policy
from gfootball.env import football_action_set
from gfootball.env.wrappers import Simple115StateWrapper
from gfootball.env import player_base
import gym
import joblib
import numpy as np
import tensorflow.compat.v1 as tf


class Player(player_base.PlayerBase):
  """An agent loaded from PPO2 cnn model checkpoint."""

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)

    self._action_set = (env_config['action_set']
                        if 'action_set' in env_config else 'default')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)
    self._player_prefix = 'player_{}'.format(player_config['index'])
    with tf.variable_scope(self._player_prefix):
      with tf.variable_scope('ppo2_model'):
        policy_fn = build_policy(DummyEnv(self._action_set), 'mlp', num_layers=5, num_hidden=128)
        self._policy = policy_fn(nbatch=1, sess=self._sess)
    _load_variables(player_config['checkpoint'], self._sess,
                    prefix=self._player_prefix + '/')

  def __del__(self):
    self._sess.close()

  def take_action(self, observation):
    assert len(observation) == 1, 'Multiple players control is not supported'

    observation = Simple115StateWrapper.convert_observation(observation, True, True)
    action = self._policy.step(observation)[0][0]
    actions = [football_action_set.action_set_dict[self._action_set][action]]
    return actions


def _load_variables(load_path, sess, prefix='', remove_prefix=True):
  """Loads variables from checkpoint of policy trained by baselines."""

  # Forked from address below since we needed loading from different var names:
  # https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py
  variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
               if v.name.startswith(prefix)]

  loaded_params = joblib.load(load_path)
  restores = []
  for v in variables:
    v_name = v.name[len(prefix):] if remove_prefix else v.name
    restores.append(v.assign(loaded_params[v_name]))

  sess.run(restores)


class DummyEnv(object):
  # We need env object to pass to build_policy, however real environment
  # is not there yet.

  def __init__(self, action_set):
    self.action_space = gym.spaces.Discrete(
        len(football_action_set.action_set_dict[action_set]))
    self.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=[115+46], dtype=np.float32)
