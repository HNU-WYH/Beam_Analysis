from enum import Enum


class LoadType(Enum):
    q = "distributed force"
    F = "point force"
    M = "moment"


class ConstraintType(Enum):
    ROTATION = "fix the rotation"
    DISPLACEMENT = "fix the transverse displacement"
    AXIAL = "fix the displacement along the axis"


class SolvType(Enum):
    STATIC = "get the static and time-independent solution"
    DYNAMIC = "get the time-dependent solution"
    EIGEN = "get the time-dependent solution based on eigenvalue method"

class ConnectionType(Enum):
    Hinge = "Interconnection with changeable angle "
    Fix = "Interconnection with fixed angle"

def uniform_load_function(x, value = 1.0):
    return value  # Constant distributed load of 1000 N/m


def triangular_load_function(x, value = 1.0):
    return x * value


def partial_uniform_load_function(x, boundary = 0.5, value = 1.0):
    if x <= boundary:
        return value
    else:
        return 0
