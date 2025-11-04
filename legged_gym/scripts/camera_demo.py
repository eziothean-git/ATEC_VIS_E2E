# Wrapper to call the actual camera demo inside the legged_gym package
import os
import sys

# Ensure our local legged_gym package (../legged_gym) takes precedence over any globally installed one
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from legged_gym.utils.helpers import get_args
from legged_gym.scripts.camera_demo import run

if __name__ == "__main__":
    args = get_args()
    run(args)
