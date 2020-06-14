import logging
import logging.handlers

logger = logging.getLogger(name="mylogger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def set_output_file(file_name):
    handler = logging.FileHandler(filename=file_name, mode='a',
                                  encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
