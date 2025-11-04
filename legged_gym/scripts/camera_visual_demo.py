#!/usr/bin/env python3
import sys
import os

# Ensure local legged_gym takes priority
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from legged_gym.scripts.camera_visual_demo import run
from legged_gym.utils import helpers

if __name__ == '__main__':
    args = helpers.get_args()
    run(args)
