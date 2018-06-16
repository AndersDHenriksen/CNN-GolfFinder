from base.base_model import BaseModel
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout

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

        # CONV -> BN -> RELU Block -> Max Pool applied to X
        X = Conv2D(32, (9, 9), strides=(5, 5), name='conv0')(self.x)
        # X = BatchNormalization(name='bn0')(X) #TODO try tf.Keras batchnormalization layer instead
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), name='max_pool0')(X)

        # CONV -> BN -> RELU Block -> Max Pool applied to X
        X = Conv2D(64, (3, 3), strides=(1, 1), name='conv1')(X)
        # X = BatchNormalization(name='bn1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), name='max_pool1')(X)

        X = Flatten()(X)
        X = Activation('relu')(X)

        X = Dense(128)(X)
        X = Activation('relu')(X)
        # X = Dropout(0.5)(X)

        # X = Dense(16)(X)
        # X = Activation('relu')(X)
        # X = Dropout(0.5)(X)

        self.y_out = Dense(2)(X)


        with tf.name_scope("loss"):
            self.squared_error = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.y_out))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.squared_error,
                                                                                         global_step=self.global_step_tensor)
            distance_to_annotation = tf.linalg.norm(self.y_out - self.y, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(distance_to_annotation, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

