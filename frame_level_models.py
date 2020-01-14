# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a collection of models which operate on variable-length sequences."""
import math

import model_utils as utils
import models
import tensorflow as tf
from tensorflow import flags
import tensorflow.contrib.slim as slim
import video_level_models

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30, "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string(
    "dbof_pooling_method", "max",
    "The pooling method used in the DBoF cluster layer. "
    "Choices are 'average' and 'max'.")
flags.DEFINE_string(
    "dbof_activation", "sigmoid",
    "The nonlinear activation method for cluster and hidden dense layer, e.g., "
    "sigmoid, relu6, etc.")
flags.DEFINE_string(
    "video_level_classifier_model", "MoeModel",
    "Some Frame-Level models can be decomposed into a "
    "generalized pooling operation followed by a "
    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")



flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the FV cluster layer.")
flags.DEFINE_integer("fv_hidden_size", 512,
                     "Number of units in the FV hidden layer.")

class FrameLevelLogisticModel(models.BaseModel):
  """Creates a logistic classifier over the aggregated frame-level features."""

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """See base class.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(tf.tile(num_frames, [1, feature_size]),
                              [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input, axis=[1]) / denominators

    output = slim.fully_connected(avg_pooled,
                                  vocab_size,
                                  activation_fn=tf.nn.sigmoid,
                                  weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}


class NeXtFVModel(models.BaseModel):
  """Creates a NeXtFV based model.
     It emulates a Gaussian Mixture Fisher Vector pooling operations
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    act_fn = self.ACT_FN_MAP.get(FLAGS.dbof_activation)

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]

    reshaped_input = tf.reshape(model_input, [-1, feature_size])

    #
    video_NeXt = NeXt(1024, max_frames, cluster_size, is_training)
    audio_NeXt = NeXt(128, max_frames, cluster_size / 2, is_training)

    reshaped_input = slim.batch_norm(reshaped_input,
                                      center=True,
                                      scale=True,
                                      is_training=is_training,
                                      scope="input_bn")

    with tf.variable_scope("video_FV"):
        fv_video = video_NeXt.forward(reshaped_input[:,0:1024]) 
    with tf.variable_scope("audio_FV"):
        fv_audio = audio_NeXt.forward(reshaped_input[:,1024:])

    fv = tf.concat([fv_video, fv_audio], 1)
    fv_dim = fv.get_shape().as_list()[1] 

    lhs_weights = tf.get_variable("lhs_weights",
        [fv_dim, hidden1_size], dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=1 /math.sqrt(cluster_size)))
    rhs_weights = tf.get_variable("rhs_weights",
        [fv_dim, hidden1_size], dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=1/math.sqrt(cluster_size)))

    lhs_elements = tf.matmul(fv, lhs_weights)
    rhs_elements = tf.matmul(fv, rhs_weights)

    lhs_elements = slim.batch_norm(
        lhs_elements,
        center=True,
        scale=True,
        is_training=is_training,
        scope="lhs_bn",
        fused=False)
    rhs_elements = slim.batch_norm(
        rhs_elements,
        center=True,
        scale=True,
        is_training=is_training,
        scope="rhs_bn",
        fused=False)

    connected = tf.add(lhs_elements, rhs_elements)
    connected = tf.mean(connected, 0)
    connected = tf.reshape(connected, [1, hidden1_size])

    lhs_squeeze_weights = tf.get_variable("lhs_squeeze_weights",
        [hidden1_size, hidden1_size], dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=1 /math.sqrt(cluster_size)))
    rhs_squeeze_weights = tf.get_variable("rhs_squeeze_weights",
        [hidden1_size, hidden1_size], dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=1/math.sqrt(cluster_size)))

    lhs_squeeze_elements = tf.matmul(connected, lhs_squeeze_weights)
    rhs_squeeze_elements = tf.matmul(connected, rhs_squeeze_weights)

    lhs_squeeze_elements = tf.nn.softmax(lhs_squeeze_elements)
    rhs_squeeze_elements = tf.nn.softmax(rhs_squeeze_elements)

    lhs_final = tf.multiply(lhs_elements, lhs_squeeze_elements)
    rhs_final = tf.multiply(rhs_elements, rhs_squeeze_elements)

    activation = tf.add(lhs_final, rhs_final)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)

class NeXtFV():
    def __init__(self, feature_size, max_frames, cluster_size, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        cluster_weights0 = tf.get_variable("cluster_weights0",
          [self.feature_size, self.cluster_size], dtype=tf.float32,
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        cluster_weights1 = tf.get_variable("cluster_weights1",
          [self.feature_size, self.cluster_size], dtype=tf.float32,
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights0)
        activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=self.is_training,
          scope="cluster_bn")
        activation = tf.nn.softmax(activation)
        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)


        attention = tf.matmul(reshaped_input, cluster_weights1)
        attention = slim.batch_norm(
          attention,
          center=True,
          scale=True,
          is_training=self.is_training,
          scope="cluster_bn")
        attention = tf.nn.swish(attention)
        attention = tf.reshape(attention, [-1, self.max_frames, self.cluster_size])
        
        b_sum = tf.reduce_sum(attention, -2, keep_dims=True)
        a_sum = tf.multiply(a_sum, b_sum)

        cluster_weights2 = tf.get_variable("cluster_weights2",
          [1, self.feature_size, self.cluster_size], dtype=tf.float32,
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        a = tf.multiply(a_sum,cluster_weights2)
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        fv1 = tf.matmul(activation, reshaped_input)
        fv1 = tf.transpose(fv1,perm=[0,2,1])

        # computing second order FV
        a2 = tf.multiply(a_sum,tf.square(cluster_weights2)) 

        b2 = tf.multiply(fv1,cluster_weights2)
        fv2 = tf.matmul(activation,tf.square(reshaped_input)) 
     
        fv2 = tf.transpose(fv2,perm=[0,2,1])
        fv2 = tf.add_n([a2,fv2,tf.scalar_mul(-2, b2)])

        covar_weights = tf.get_variable("covar_weights",
          [self.feature_size, self.cluster_size], dtype=tf.float32,
          initializer = tf.random_normal_initializer(mean=1.000001, stddev=2 /math.sqrt(self.feature_size)))

        fv2 = tf.divide(fv2,tf.square(covar_weights))
        fv2 = tf.subtract(fv2,a_sum)
        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2,1)

        fv1 = tf.subtract(fv1, a)
        fv1 = tf.divide(fv1,covar_weights) 
        fv1 = tf.reshape(fv1,[-1,self.cluster_size*self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1,1)

        return tf.concat([fv1,fv2], 1)




class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  """

  ACT_FN_MAP = {
      "sigmoid": tf.nn.sigmoid,
      "relu6": tf.nn.relu6,
  }

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    """See base class.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).
      iterations: the number of frames to be sampled.
      add_batch_norm: whether to add batch norm during training.
      sample_random_frames: whether to sample random frames or random sequences.
      cluster_size: the output neuron number of the cluster layer.
      hidden_size: the output neuron number of the hidden layer.
      is_training: whether to build the graph in training mode.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    act_fn = self.ACT_FN_MAP.get(FLAGS.dbof_activation)
    assert act_fn is not None, ("dbof_activation is not valid: %s." %
                                FLAGS.dbof_activation)

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.compat.v1.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(reshaped_input,
                                       center=True,
                                       scale=True,
                                       is_training=is_training,
                                       scope="input_bn")

    cluster_weights = tf.compat.v1.get_variable(
        "cluster_weights", [feature_size, cluster_size],
        initializer=tf.random_normal_initializer(stddev=1 /
                                                 math.sqrt(feature_size)))
    tf.compat.v1.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(activation,
                                   center=True,
                                   scale=True,
                                   is_training=is_training,
                                   scope="cluster_bn")
    else:
      cluster_biases = tf.compat.v1.get_variable(
          "cluster_biases", [cluster_size],
          initializer=tf.random_normal_initializer(stddev=1 /
                                                   math.sqrt(feature_size)))
      tf.compat.v1.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = act_fn(activation)
    tf.compat.v1.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.compat.v1.get_variable(
        "hidden1_weights", [cluster_size, hidden1_size],
        initializer=tf.random_normal_initializer(stddev=1 /
                                                 math.sqrt(cluster_size)))
    tf.compat.v1.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(activation,
                                   center=True,
                                   scale=True,
                                   is_training=is_training,
                                   scope="hidden1_bn")
    else:
      hidden1_biases = tf.compat.v1.get_variable(
          "hidden1_biases", [hidden1_size],
          initializer=tf.random_normal_initializer(stddev=0.01))
      tf.compat.v1.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = act_fn(activation)
    tf.compat.v1.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(model_input=activation,
                                           vocab_size=vocab_size,
                                           **unused_params)


class LstmModel(models.BaseModel):
  """Creates a model which uses a stack of LSTMs to represent the video."""

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """See base class.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
        for _ in range(number_of_layers)
    ])

    _, state = tf.nn.dynamic_rnn(stacked_lstm,
                                 model_input,
                                 sequence_length=num_frames,
                                 dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(model_input=state[-1].h,
                                           vocab_size=vocab_size,
                                           **unused_params)
