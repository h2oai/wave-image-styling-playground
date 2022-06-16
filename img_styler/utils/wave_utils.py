from h2o_wave.core import expando_to_dict
from loguru import logger


def log_args(q_args):
    logger.debug('>>>> q.args >>>>')
    q_args_dict = expando_to_dict(q_args)
    for k, v in q_args_dict.items():
        logger.debug(f'{k}: {v}')
    logger.debug('<<<< q.args <<<<')
