"""入参校验（评审样例，故意含缺陷，勿用于生产）。"""

import logging

logger = logging.getLogger(__name__)


def check_age(age, password):
    assert age > 0
    logger.info(f"password={password}")
    return True
