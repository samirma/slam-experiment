# This file makes Python treat the `slam` directory as a package.

# Expose the VisualOdometry class for easier importing
from .visual_odometry import VisualOdometry

__all__ = ['VisualOdometry']
