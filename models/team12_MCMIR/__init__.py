import os
import sys

# Add the current directory to sys.path if it's not already there.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if os.path.join(current_dir,'basicsr') not in sys.path:
    sys.path.insert(0, os.path.join(current_dir,'basicsr'))

from .basicsr.test import *