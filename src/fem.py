import numpy as np
from beam import Beam
from utils.local_matrix import LocalElement, LoadType


class FEM:
    def __init__(self, beam: Beam):
        self.beam = beam
        self.S = self.beam.S
        self.M = self.beam.M
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
                raise Warning("force applied beyond the beam", f_pos)
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

    def apply_constraint(self):
        pass
