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
    log_generic_to_tensorboard(writer, step, prefix, "loss", loss)
    log_generic_to_tensorboard(writer, step, prefix, "accuracy", accuracy)

def log_generic_to_tensorboard(writer, step, prefix, metric, value):
    writer.add_scalar("{}/{}".format(prefix, metric), value)