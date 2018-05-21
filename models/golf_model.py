from base.base_model import BaseModel
import tensorflow as tf


class GolfBallModel(BaseModel):
    def __init__(self, config):
        super(GolfBallModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

        # network architecture
        x_flatten = tf.layers.flatten(self.x[..., -1])
        d1 = tf.layers.dense(x_flatten, 128, activation=tf.nn.relu, name="dense1")
        self.y_out = tf.layers.dense(d1, 2, name="dense2")

        with tf.name_scope("loss"):
            self.squared_error = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.y_out))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.squared_error,
                                                                                         global_step=self.global_step_tensor)
            distance_to_annotation = tf.linalg.norm(self.y_out - self.y)
            self.accuracy = tf.reduce_mean(tf.cast(distance_to_annotation, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

