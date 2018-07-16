import tensorflow as tf
from tensorflow.python.keras import backend as K

from data_loader.data_generator import DataGenerator
from models.golf_model import GolfBallModel
from trainers.golf_trainer import GolfBallTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from utils.golf_predict import GolfBallPrediction


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create your data generator
    data = DataGenerator(config)
    # create tensorflow session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    K.set_session(sess)
    # create an instance of the model you want
    model = GolfBallModel(config, data)
    # load model if exists
    model.load(sess)
    # create tensorboard logger
    logger = Logger(sess, config)
    if config.do_training:
        # create trainer and pass all the previous components to it
        trainer = GolfBallTrainer(sess, model, data, config, logger)
        # here you train your model
        trainer.train()
    if config.do_predict:
        GolfBallPrediction(sess, model, data, config)


if __name__ == '__main__':
    main()
    # Run with:
    # -- script_path: /home/anders/PycharmProjects/BallFinder/mains/golf_main.py
    # --  parameters: --config=./configs/golf_config.json
    # -- working dir: /home/anders/PycharmProjects/BallFinder
    # (linux) To start tensorboard in console run: tensorboard --logdir=../experiments
    # (windows) To start tensorboard in conda console run:
    # > conda activate tensorflow
    # > tensorboard --logdir="C:\Users\ahe\Google Drive\TrackMan\01. FullSwing\DeepLearning\experiments"
