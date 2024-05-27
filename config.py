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
