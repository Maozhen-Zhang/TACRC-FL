import argparse
import logging
import os

import yaml
logger = logging.getLogger("logger")


def get_conf(current_time):
    parser = argparse.ArgumentParser(description="Federated Learning")
    parser.add_argument('-c', '--conf', dest='conf', default='./conf/conf.yaml', help="config file path")
    args = parser.parse_args()
    with open(args.conf,"r+") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

    logPath = f"{conf['logPath'] + current_time}"
    try:
        os.mkdir(conf['logPath'])
    except FileExistsError:
        logger.info(f"Folder {conf['logPath']} already exists")

    try:
        os.mkdir(logPath)
    except FileExistsError:
        logger.info(f"Folder {logPath} already exists")


    logger.addHandler(logging.FileHandler(filename=f'./{logPath}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    with open(f'{logPath}/params.yaml', 'w') as f:
        yaml.dump(conf, f)
    return conf
