from enum import Enum


class LoadType(Enum):
    q = "distributed force"
    F = "point force"
    M = "moment"


class ConstraintType(Enum):
    ROTATION = "fix the rotation"
    DISPLACEMENT = "fix the displacement"


class SolvType(Enum):
    STATIC = "get the static and time-independent solution"
    DYNAMIC = "get the time-dependent solution"

class ConnectionType(Enum):
    Hinge = "Interconnection with changeable angle "
    Fix = "Interconnection with fixed angle"

def uniform_load_function(x):
    return 1.0  # Constant distributed load of 1000 N/m


def triangular_load_function(x):
    return x


def partial_uniform_load_function(x):
    if x <= 0.5:
        return 1
    else:
        return 0
