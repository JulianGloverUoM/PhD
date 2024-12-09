# For running code on the maths cs servers, will figure out how to do shit properly later

######################################
import os
import sys
import pickle
import random
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import scipy as sp
import scipy.stats as stats
from datetime import date
import copy
import time

file_path = os.path.realpath(__file__)
sys.path.append(file_path)
# import Biaxial_stretch_Radau  # noqa
# import Stretch_Radau  # noqa
# import PBC_network  # noqa
# import Dilation_radau  # noqa
import Create_PBC_Network  # noqa
import Dilation_PBC  # noqa

######################################

L = sys.argv[1]
density = sys.argv[2]
