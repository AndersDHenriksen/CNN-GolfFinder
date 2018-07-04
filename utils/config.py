import json
from munch import Munch
from pathlib import Path
from datetime import datetime
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Munch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    experiment_folder = Path("../experiments")
    do_load_exp = "_run" in config.exp_name
    if do_load_exp:
        exp_name = list(experiment_folder.glob('*' + config.exp_name))
        if len(exp_name):
            config.exp_name = exp_name[-1].stem
        else:
            do_load_exp = False
    if not do_load_exp:
        run_n = len(list(experiment_folder.glob('*' + config.exp_name + '*')))
        config.exp_name = "{} - {}_run{}".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S'), config.exp_name, run_n)
    config.summary_dir = "{}".format(os.path.realpath(str(experiment_folder / config.exp_name / "summary")) + os.sep)
    config.checkpoint_dir = "{}".format(os.path.realpath(str(experiment_folder / config.exp_name / "checkpoint")) + os.sep)

    return config
