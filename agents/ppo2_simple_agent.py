import os
import collections
import gym
import numpy as np
import joblib
import tensorflow.compat.v1 as tf
from baselines.common.policies import build_policy
from gfootball.env import football_action_set
from gfootball.env import player_base
from gfootball.env.wrappers import Simple115StateWrapper


class Player(player_base.PlayerBase):
  """An agent loaded from PPO2 cnn model checkpoint."""

  def __init__(self, checkpoint_path):
    player_base.PlayerBase.__init__(self)
    self._action_set = 'default'
    self._player_prefix = 'player_0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)
    with tf.variable_scope(self._player_prefix):
      with tf.variable_scope('ppo2_model'):
        policy_fn = build_policy(DummyEnv(self._action_set), 'mlp', num_layers=5, num_hidden=128)
        self._policy = policy_fn(nbatch=1, sess=self._sess)
    _load_variables(checkpoint_path, self._sess, prefix=self._player_prefix + '/')
    saver = tf.train.Saver()
    saver.save(self._sess, "/home/alex/Dropbox/projects/python/kaggle/football/saved_models/simple_ppo2/simple_ppo2")

  def __del__(self):
    self._sess.close()

  def take_action(self, observation):
    assert len(observation) == 1, 'Multiple players control is not supported'

    observation = Simple115StateWrapper.convert_observation(observation, True, True)
    action = self._policy.step(observation)[0][0]
    actions = [action] #[football_action_set.action_set_dict[self._action_set][action]]
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

player = Player("/home/alex/Dropbox/projects/python/kaggle/football/checkpoints/openai-2020-11-26-12-35-02-877222/checkpoints/03600")