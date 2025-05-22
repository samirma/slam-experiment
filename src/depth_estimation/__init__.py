# This file makes Python treat the `depth_estimation` directory as a package.

# Expose the MiDaSDepthEstimator class for easier importing
from .midas import MiDaSDepthEstimator

__all__ = ['MiDaSDepthEstimator']
