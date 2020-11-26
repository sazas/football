import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from gfootball.env import observation_preprocessing
from gfootball.env import player_base


class Player(player_base.PlayerBase):
  """An agent loaded from PPO2 cnn model checkpoint."""

  def __init__(self, model_path):
    player_base.PlayerBase.__init__(self)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)
    stacking = 4
    self._stacker = ObservationStacker(stacking)
    imported_graph = tf.train.import_meta_graph(model_path + '.meta')
    imported_graph.restore(self._sess, model_path)

  def __del__(self):
    self._sess.close()

  def take_action(self, observation):
    observation = observation_preprocessing.generate_smm(observation)
    observation = self._stacker.get(observation)
    action = self._sess.run("player_0/ppo2_model/ArgMax:0", feed_dict={"player_0/ppo2_model/Ob:0": observation})
    return [int(action[0])]

  def reset(self):
    self._stacker.reset()


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
