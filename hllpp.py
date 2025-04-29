import math
import hashlib
import numpy as np
import xxhash
import copy

class HyperLogLogPlusPlus:
    def __init__(self, p=14):
        """
        p: precision parameter (2^p registers)
        """
        self.p = p
        self.m = 2 ** p
        self.alpha = self._get_alpha()
        self.registers = np.zeros(self.m, dtype=int) # buckets
        self.sparse_list = []
        self.sparse_threshold = 1600  # threshold for switching to dense mode
        self.p_prime = 25 # sparse encoding uses up to 2^p'
        self.tmp_set = set()
        self.m_prime = 2 ** self.p_prime
        self.mode = 'sparse'

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
        # h = hashlib.(value.encode('utf-8')).hexdigest()
        h = xxhash.xxh64(value.encode('utf-8')).hexdigest()
        return int(h, 16)

    def add(self, value):
        x = self._hash(value)
        if self.mode == 'sparse':
            # print("adding in sparse mode")
            k = self._encode_sparse(x)
            self.tmp_set.add(k)

            if len(self.tmp_set) > 512:  # some buffer before merging
                self._merge_tmp_set()
                self.tmp_set = set()
                # Check if sparse representation exceeds size
                # if len(self.sparse_list) > self.m * 6:  # assume 12 bits per entry
                if len(self.sparse_list) > self.sparse_threshold:
                    self._convert_sparse_to_dense()
        else:
            # print("adding in dense mode")
            self._update_registers(x)

    def _encode_sparse(self, x):
        idx = x >> (64 - self.p)
        w = x & ((1 << (64 - self.p)) - 1)
        rank = self._rank(w, 64 - self.p)
        return (idx << 6) | rank  # 6 bits to store rank (max 64)
    
    def _merge_tmp_set(self):
        self.sparse_list = sorted(set(self.sparse_list).union(self.tmp_set))
        self.tmp_set.clear()


    def _rank(self, bits, max_bits):
        """Count leading zeros in bits."""
        return max_bits - bits.bit_length() + 1

    def _linear_counting(self, m, V):
        # print("in linear count: m=", m, ", V=", V)
        return m * math.log(m / V) # V is number of zero-valued buckets
    
    def _convert_sparse_to_dense(self):
        for encoded in self.sparse_list:
            idx = encoded >> 6
            rank = encoded & 0x3F
            self.registers[idx] = max(self.registers[idx], rank)
        self.sparse_list = []
        self.tmp_set = set()
        self.mode = 'dense'

    def _update_registers(self, x):
        idx = x >> (64 - self.p)  # first p bits
        # print(bin(x)[:15], bin(idx))
        w = x & ((1 << (64 - self.p)) - 1)  # just maintain the lower 64-p bits of the hash
        # print(bin(w))
        # print(bin(x)[-15:], bin(w)[-15:])
        # print(len(bin(x)), len(bin(w)))

        rank = self._rank(w, 64 - self.p) # number of leading zeros
        # rank = self.count_leading_zeros(w)
        # print("num leading zeros: ", rank, ", idx: ", idx)
        self.registers[idx] = max(self.registers[idx], rank) # update max in bucket

    def estimate(self):
        if self.mode == 'sparse':
            self._merge_tmp_set()
            V = self.m_prime - len(self.sparse_list)
            # V = self.m - np.count_nonzero(self.registers)
            # print("V: ", V)
            return self._linear_counting(self.m_prime, V)
        Z = 1.0 / np.sum(2.0 ** -self.registers)
        E = self.alpha * self.m * self.m * Z
        # print("E in estimate: ", E)

        # Bias correction and thresholds
        if E <= 2.5 * self.m: # small cardinality
            V = np.count_nonzero(self.registers == 0)
            if V != 0:
                return self._linear_counting(self.m, V)
        elif E > (1/30) * (2 ** 32): # extremely large cardinality
            return -(2 ** 32) * math.log(1 - (E / 2 ** 32))
        return E # default

    def aggregate(self, other):
        if not isinstance(other, HyperLogLogPlusPlus):
            raise TypeError("Can only merge with another HLL++")
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different precision")
        if self.mode == 'sparse' and other.mode == 'sparse':
            self.sparse_list = self.merge_sparse_lists(other.sparse_list)
        else:
            self._convert_sparse_to_dense()
            copy = copy.deepcopy(other)
            copy._convert_sparse_to_dense()
            self.registers = np.maximum(self.registers.copy(), copy.registers.copy()) # bucket-wise maximums
        if len(self.sparse_list) > self.sparse_threshold:
            self._convert_sparse_to_dense()
            self.registers = np.maximum(self.registers.copy(), other.registers.copy()) # bucket-wise maximums
    
    def merge_sparse_lists(self, other_sparse_list):
        merged = {}

        for encoded in self.sparse_list + other_sparse_list:
            idx = encoded >> 6
            rank = encoded & 0x3F  # 6 bits for rank

            if idx not in merged or rank > merged[idx]:
                merged[idx] = rank

        # Re-encode: (idx << 6) | rank
        return [(idx << 6) | rank for idx, rank in merged.items()]

