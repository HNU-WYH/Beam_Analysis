from .beam import Beam2D, Beam
from .config import LoadType, ConstraintType, SolvType, ConnectionType, uniform_load_function, triangular_load_function, \
    partial_uniform_load_function
from .fem_beam import FEM
from .fem_frame import FrameworkFEM

LEFT_END = 0
RIGHT_END = -1
