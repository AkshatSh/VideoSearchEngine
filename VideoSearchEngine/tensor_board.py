from tensorboard import SummaryWriter

from constants import (
    LOG_DIR
)

def create_writer():
    return SummaryWriter(LOG_DIR)

def log_to_tensorboard(writer, step, prefix, loss, accuracy):
    """
    Log metrics to Tensorboard.
    """
    writer.add_scalar("{}/loss".format(prefix), loss, step)
    writer.add_scalar("{}/accuracy".format(prefix),
                      accuracy, step)
