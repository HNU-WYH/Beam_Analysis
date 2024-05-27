from enum import Enum


class LoadType(Enum):
    p = "distributed force"
    F = "point force"
    M = "moment"


class ConstraintType(Enum):
    rotation = "fix the rotation"
    displacement = "fix the displacement"


class SolvType(Enum):
    static = "get the static and time-independent solution"
    dynamic = "get the time-dependent solution"
