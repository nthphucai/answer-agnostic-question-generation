from loguru import logger
from pymongo import MongoClient
from pymongo.errors import OperationFailure

from questgen.utils.file_utils import read_yaml_file


def connect_mongodb(config_path: str, collection_name: str):
    config = read_yaml_file(config_path)

    user = config["mongodb"]["user"]
    password = config["mongodb"]["password"]
    host = config["mongodb"]["host"]
    port = config["mongodb"]["port"]
    myclient = MongoClient(
        "mongodb://"
        + user
        + ":"
        + password
        + "@"
        + host
        + ":"
        + port
        + "/?authMechanism=DEFAULT"
    )
    try:
        logger.info(myclient.server_info())
        db = myclient[config["mongodb"]["database_name"]]
        collection = db[config["mongodb"][collection_name]]
        return collection
    except OperationFailure as e:
        logger.error(e)
        return None
