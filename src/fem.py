import numpy as np
from src.beam import Beam
from utils.config import LoadType, ConstraintType, SolvType
from utils.local_matrix import LocalElement
from utils.newmark import NewMark


class FEM:
    def __init__(self, beam: Beam):
        self.S = None
        self.M = None
        self.stsol = None
        self.dysol = None
        self.beam = beam
        self.constraints = []
        self.q = np.zeros(2 * beam.num_nodes)

    def apply_force(self, load, load_type):
        if load_type == LoadType.p:
            for idx, xstart in enumerate(self.beam.nodes[0:-1]):
                self.q[2 * idx:2 * idx + 4] += LocalElement.equal_force(
                    load,
                    load_type,
                    xstart,
                    self.beam.element_len
                )

        elif load_type == LoadType.F:
            f_pos, f_val = load
            idx = int(f_pos / self.beam.element_len)
            if f_pos > self.beam.L or f_pos < 0:
                Warning("force applied beyond the beam", f_pos)
            elif f_pos in self.beam.nodes:
                self.q[2 * idx] = f_val
            else:

                self.q[2 * idx:2 * idx + 4] += LocalElement.equal_force(
                    load,
                    load_type,
                    idx * self.beam.element_len,
                    self.beam.element_len
                )

        elif load_type == LoadType.M:
            m_pos, m_val = load
            idx = int(m_pos / self.beam.element_len)
            if m_pos > self.beam.L or m_pos < 0:
                raise Warning("moment applied beyond the beam", m_pos)
            elif m_pos in self.beam.nodes:
                self.q[2 * idx + 1] = m_val
            else:
                self.q[2 * idx:2 * idx + 4] += LocalElement.equal_force(
                    load,
                    load_type,
                    idx * self.beam.element_len,
                    self.beam.element_len
                )

    def add_constraint(self, node, value, constraint_Type: ConstraintType):
        self.constraints.append((node, value, constraint_Type))

    def __apply_constraint(self):
        num_constraints = len(self.constraints)
        original_size = self.beam.S.shape[0]
        expand_size = num_constraints + original_size

        # create the expanded matrix
        self.S = np.zeros((expand_size, expand_size))
        self.M = np.zeros((expand_size, expand_size))
        self.new_q = np.zeros(expand_size)

        # copy original S,M,q into expanded matrix
        self.S[0:original_size,0:original_size] = self.beam.S
        self.M[0:original_size, 0:original_size] = self.beam.M
        self.new_q[0:original_size] = self.q

        # add_constraints
        for i, (node, value, constraint_type) in enumerate(self.constraints):
            constraint_idx  = original_size+i
            self.new_q[constraint_idx] = value

            if constraint_type == ConstraintType.rotation:
                self.S[constraint_idx, 2 * node + 1] = 1
                self.S[2 * node + 1, constraint_idx] = 1
            elif constraint_type == ConstraintType.displacement:
                self.S[constraint_idx, 2 * node] = 1
                self.S[2 * node, constraint_idx] = 1
            else:
                Warning("wrong type of constraint", constraint_type)

    def solv(self, num_steps=None, tau=None, soltype=SolvType.static, beta=0.25, gamma=0.5):
        self.__apply_constraint()
        if soltype == SolvType.static:
            self.stsol = np.linalg.solve(self.S, self.new_q)
        elif soltype == SolvType.dynamic:
            newmark_solver = NewMark(tau, num_steps, beta, gamma)
            init_condis = np.zeros(self.new_q.shape)
            self.dysol, _, _ = newmark_solver.solve(self.M, self.S, self.new_q, init_condis, init_condis, init_condis)
        else:
            raise Exception("Wrong defined type of solution")
