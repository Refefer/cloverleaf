import os
import sys

# Lets the tests import ep_gpu from the source tree.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
