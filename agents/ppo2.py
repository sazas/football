import os
import collections
import gym
import numpy as np
import joblib
import tensorflow.compat.v1 as tf
import sonnet as snt
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.policies import PolicyWithValue
from gfootball.env import football_action_set
from gfootball.env import observation_preprocessing
from gfootball.env import player_base
from gfootball.env import wrappers


def gfootball_impala_cnn_network_fn(frame):
    # Convert to floats.
    frame = tf.to_float(frame)
    frame /= 255
    with tf.variable_scope('convnet'):
        conv_out = frame
        conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
        for i, (num_ch, num_blocks) in enumerate(conv_layers):
            # Downscale.
            conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            conv_out = tf.nn.pool(
                conv_out,
                window_shape=[3, 3],
                pooling_type='MAX',
                padding='SAME',
                strides=[2, 2])

            # Residual block(s).
            for j in range(num_blocks):
                with tf.variable_scope('residual_%d_%d' % (i, j)):
                    block_input = conv_out
                    conv_out = tf.nn.relu(conv_out)
                    conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                    conv_out = tf.nn.relu(conv_out)
                    conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                    conv_out += block_input

    conv_out = tf.nn.relu(conv_out)
    conv_out = snt.BatchFlatten()(conv_out)

    conv_out = snt.Linear(256)(conv_out)
    conv_out = tf.nn.relu(conv_out)

    return conv_out


class PlayerBase(object):
  """Base player class."""

  def __init__(self):
    self._num_left_controlled_players = 1
    self._num_right_controlled_players = 0
    self._can_play_right = False
    
  def num_controlled_left_players(self):
    return self._num_left_controlled_players

  def num_controlled_right_players(self):
    return self._num_right_controlled_players

  def num_controlled_players(self):
    return (self._num_left_controlled_players +
            self._num_right_controlled_players)

  def reset(self):
    pass

  def can_play_right(self):
    return self._can_play_right


class Player(player_base.PlayerBase):
  """An agent loaded from PPO2 cnn model checkpoint."""

  def __init__(self, checkpoint_path):
    player_base.PlayerBase.__init__(self)
    self._action_set = 'default'
    self._player_prefix = 'player_0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)
    stacking = 4
    self._stacker = ObservationStacker(stacking)
    with tf.variable_scope(self._player_prefix):
        with tf.variable_scope('ppo2_model'):
            env = DummyEnv(self._action_set, stacking)
            ob_space = env.observation_space
            X = observation_placeholder(ob_space, batch_size=1)
            extra_tensors = {}
            encoded_x = X
            encoded_x = encode_observation(ob_space, encoded_x)
            with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
                policy_latent = gfootball_impala_cnn_network_fn(encoded_x)
            self._policy = PolicyWithValue(
                env=env,
                observations=X,
                latent=policy_latent,
                vf_latent=policy_latent,
                sess=self._sess,
                estimate_q=False,
                **extra_tensors
            )
    _load_variables(checkpoint_path, self._sess, prefix=self._player_prefix + '/')
    saver = tf.train.Saver()
    saver.save(self._sess, "/home/alex/Dropbox/projects/python/kaggle/football/saved_models/11_vs_11_easy_stochastic_v2")

  def __del__(self):
    self._sess.close()

  def take_action(self, observation):
    assert len(observation) == 1, 'Multiple players control is not supported'

    observation = observation_preprocessing.generate_smm(observation)
    observation = self._stacker.get(observation)
    action = self._policy.step(observation)[0][0]
    actions = [action] #[football_action_set.action_set_dict[self._action_set][action]]
    return actions

  def reset(self):
    self._stacker.reset()


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


class ObservationStacker(object):
  """Utility class that produces stacked observations."""

  def __init__(self, stacking):
    self._stacking = stacking
    self._data = []

  def get(self, observation):
    if self._data:
      self._data.append(observation)
      self._data = self._data[-self._stacking:]
    else:
      self._data = [observation] * self._stacking
    return np.concatenate(self._data, axis=-1)

  def reset(self):
    self._data = []


class DummyEnv(object):
  # We need env object to pass to build_policy, however real environment
  # is not there yet.

  def __init__(self, action_set, stacking):
    self.action_space = gym.spaces.Discrete(
        len(football_action_set.action_set_dict[action_set]))
    self.observation_space = gym.spaces.Box(
        0, 255, shape=[72, 96, 4 * stacking], dtype=np.uint8)