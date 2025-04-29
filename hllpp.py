import math
import hashlib
import numpy as np

class HyperLogLogPlusPlus:
    def __init__(self, p=14):
        """
        p: precision parameter (2^p registers)
        """
        self.p = p
        self.m = 2 ** p
        self.alpha = self._get_alpha()
        self.registers = np.zeros(self.m, dtype=int) # buckets
        self.sparse_list = set()
        self.sparse_threshold = 2000  # threshold for switching to dense mode

    def _get_alpha(self):
        if self.m == 16:
            return 0.673
        elif self.m == 32:
            return 0.697
        elif self.m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / self.m)

    def _hash(self, value):
        """Hash using SHA256 then interpret the result as integer."""
        h = hashlib.sha256(value.encode('utf-8')).hexdigest()
        return int(h, 16)

    def add(self, value):
        x = self._hash(value)
        if len(self.sparse_list) < self.sparse_threshold:
            self.sparse_list.add(x)
        else:
            idx = x >> (256 - self.p)  # first p bits
            w = (x << self.p) & ((1 << 256) - 1)  # just maintain the lower 256-p bits of the hash
            rank = self._rank(w, 256 - self.p) # number of leading zeros
            self.registers[idx] = max(self.registers[idx], rank) # update max in bucket

    def _rank(self, bits, max_bits):
        """Count leading zeros in bits."""
        return max_bits - bits.bit_length() + 1

    def _linear_counting(self, V):
        return self.m * math.log(self.m / V) # V is number of zero-valued buckets

    def estimate(self):
        if len(self.sparse_list) < self.sparse_threshold:
            V = self.m - np.count_nonzero(self.registers)
            return self._linear_counting(V)
        Z = 1.0 / np.sum(2.0 ** -self.registers)
        E = self.alpha * self.m * self.m * Z

        # Bias correction and thresholds
        if E <= 2.5 * self.m: # small cardinality
            V = np.count_nonzero(self.registers == 0)
            if V != 0:
                return self._linear_counting(V)
        elif E > (1/30) * (2 ** 32): # extremely large cardinality
            return -(2 ** 32) * math.log(1 - (E / 2 ** 32))
        return E # default

    def merge(self, other):
        if not isinstance(other, HyperLogLogPlusPlus):
            raise TypeError("Can only merge with another HLL++")
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different precision")
        self.registers = np.maximum(self.registers, other.registers) # bucket-wise maximums
    
    def sendRegisters(self):
        return self.registers.copy()
