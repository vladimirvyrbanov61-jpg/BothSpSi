"""Neural distinguisher models and training."""

from thesis.models.cnn_distinguisher import CnnDistinguisher
from thesis.models.train import TrainConfig, train_distinguisher

__all__ = ["CnnDistinguisher", "TrainConfig", "train_distinguisher"]
