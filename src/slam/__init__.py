# This file makes Python treat the `slam` directory as a package.

# Expose the SLAMFrontend class for easier importing
from .slam_frontend import SLAMFrontend

__all__ = ['SLAMFrontend']
