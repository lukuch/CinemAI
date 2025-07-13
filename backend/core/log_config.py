import logging

import structlog


def setup_logging(dev_mode=True):
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    pre_chain = [
        structlog.stdlib.add_log_level,
        timestamper,
    ]
    if dev_mode:
        processors = pre_chain + [structlog.dev.ConsoleRenderer()]
    else:
        processors = pre_chain + [structlog.processors.JSONRenderer()]
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=logging.INFO,
    )
