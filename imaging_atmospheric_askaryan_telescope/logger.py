import time
import logging
import sys
import io


DATEFMT_ISO8601 = "%Y-%m-%dT%H:%M:%S"
FMT = "%(asctime)s.%(msecs)03d"
FMT += " "
FMT += "%(levelname)s"
FMT += " "
FMT += "%(message)s"


def LoggerStdout(name="stdout"):
    return LoggerStream(stream=sys.stdout, name=name)


def LoggerStdout_if_logger_is_None(logger):
    if logger is None:
        return LoggerStdout()
    else:
        return logger


def LoggerStream(stream=sys.stdout, name="stream"):
    lggr = logging.Logger(name=name)
    fmtr = logging.Formatter(fmt=FMT, datefmt=DATEFMT_ISO8601)
    stha = logging.StreamHandler(stream)
    stha.setFormatter(fmtr)
    lggr.addHandler(stha)
    lggr.setLevel(logging.DEBUG)
    return lggr


def LoggerFile(path, name="file"):
    lggr = logging.Logger(name=name)
    file_handler = logging.FileHandler(filename=path, mode="w")
    fmtr = logging.Formatter(fmt=FMT, datefmt=DATEFMT_ISO8601)
    file_handler.setFormatter(fmtr)
    lggr.addHandler(file_handler)
    lggr.setLevel(logging.DEBUG)
    return lggr


def shutdown(logger):
    for fh in logger.handlers:
        fh.flush()
        fh.close()
        logger.removeHandler(fh)


class StartStop:
    def __init__(
        self, start_msg, logger=None, stop_msg="Done.", level=logging.DEBUG
    ):
        self.logger = LoggerStdout_if_logger_is_None(logger)
        self.level = level
        self.start_msg = start_msg
        self.stop_msg = stop_msg

    def __enter__(self):
        self.logger.log(level=self.level, msg=self.start_msg)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.logger.log(level=self.level, msg=self.stop_msg)
        return

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"
