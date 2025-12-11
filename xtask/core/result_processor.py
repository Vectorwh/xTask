import inspect
import logging


def verify_result(result):
    logging.debug("entry function: %s"% inspect.currentframe().f_code.co_name)


def write_result(result):
    logging.debug("entry function: %s"% inspect.currentframe().f_code.co_name)
