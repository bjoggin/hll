import math
import hashlib
import numpy as np
import xxhash
import copy
from collections import defaultdict

class HyperLogLogPlusPlus:
    def __init__(self, p=14, p_prime=25):
        self.mode = 'sparse'
        # for dense representation
        self.p = p
        self.m = 2 ** p
        self.alpha = self._get_alpha()
        self.registers = np.zeros(self.m, dtype=int) # buckets
        self.saved_ranks_for_k_anon_dense = [[] for _ in range(self.m)]

        # for sparse representation
        self.p_prime = p_prime # sparse encoding uses up to 2^p'
        self.m_prime = 2 ** self.p_prime
        self.sparse_list = []
        self.sparse_threshold = 1600  # threshold for switching to dense mode
        self.tmp_set = set()
        self.saved_data_for_k_anon_sparse = [[] for _ in range(self.m_prime)]

    def _get_alpha(self):
        # according to Google's paper
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
        # print(value)
        x = self._hash(value)
        if self.mode == 'sparse':
            # print("adding in sparse mode")
            k = self._encode_sparse(x) # idx and number of leading zero concatenated into 1 number
            self.tmp_set.add(k)
            # print(bin(x))
            idx, _ = self._decode_sparse_item(k)
            self.saved_data_for_k_anon_sparse[idx].append((k,x))

            if len(self.tmp_set) > 512:  # some buffer before merging
                self._merge_tmp_set()
                self.tmp_set = set()
                # Check if sparse representation exceeds size
                # if len(self.sparse_list) > self.m * 6:
                if self.exceedingSparse():
                    self._convert_sparse_to_dense()
        else:
            # print("adding in dense mode")
            self._update_registers(x)

    def _encode_sparse(self, x):
        idx = x >> (64 - self.p) # use first p number of bits to get index
        w = x & ((1 << (64 - self.p)) - 1)
        rank = self._rank(w, 64 - self.p)
        return (idx << 6) | rank
    
    def _merge_tmp_set(self):
        # self.sparse_list = sorted(set(self.sparse_list).union(self.tmp_set))
        self.sparse_list = self.merge_sparse_lists(self.tmp_set)
        self.tmp_set.clear()


    def _rank(self, bits, max_bits):
        """Count leading zeros in bits."""
        return max_bits - bits.bit_length() + 1

    def _linear_counting(self, m, V):
        # print("in linear count: m=", m, ", V=", V)
        return m * math.log(m / V) # V is number of zero-valued buckets
    
    def _convert_sparse_to_dense(self):
        self._merge_tmp_set()
        for encoded in self.sparse_list:
            idx, rank = self._decode_sparse_item(encoded)
            self.registers[idx] = max(self.registers[idx], rank)
        self.sparse_list = []
        self.tmp_set = set()
        for hashed_list in self.saved_data_for_k_anon_sparse:
            for pair in hashed_list:
                hashed = pair[1]
                idx, rank = self._encode_dense(hashed)
                self.saved_ranks_for_k_anon_dense[idx].append(rank)
        self.saved_data_for_k_anon_sparse = []
        self.mode = 'dense'

    def _update_registers(self, x):
        idx, rank = self._encode_dense(x)
        self.registers[idx] = max(self.registers[idx], rank) # update max in bucket
        self.saved_ranks_for_k_anon_dense[idx].append(rank)
    
    def _encode_dense(self, x):
        idx = x >> (64 - self.p)  # first p bits
        w = x & ((1 << (64 - self.p)) - 1)  # just maintain the lower 64-p bits of the hash
        rank = self._rank(w, 64 - self.p) # number of leading zeros
        return idx, rank


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
    
    def exceedingSparse(self):
        return len(self.sparse_list) > self.sparse_threshold
        # return len(self.sparse_list) > self.m * 6:

    def aggregate(self, other):
        if not isinstance(other, HyperLogLogPlusPlus):
            raise TypeError("Can only merge with another HLL++")
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different precision")
        if self.mode == 'sparse' and other.mode == 'sparse':
            self._merge_tmp_set()
            other._merge_tmp_set()
            self.sparse_list = self.merge_sparse_lists(other.sparse_list)
            for idx, hashed_list in enumerate(other.saved_data_for_k_anon_sparse):
                self.saved_data_for_k_anon_sparse[idx].extend(hashed_list)

        if self.mode == 'dense' or self.exceedingSparse():
            print("exceeded or dense")
            self._convert_sparse_to_dense() # does nothing if already dense
            print(self.saved_ranks_for_k_anon_dense)
            other_copy = copy.deepcopy(other)
            other_copy._convert_sparse_to_dense()
            print(other_copy.saved_ranks_for_k_anon_dense)
            self.registers = np.maximum(self.registers.copy(), other_copy.registers.copy()) # bucket-wise maximums
            for idx, ranks in enumerate(other_copy.saved_ranks_for_k_anon_dense):
                self.saved_ranks_for_k_anon_dense[idx].extend(ranks)
    
    def merge_sparse_lists(self, other_sparse_list):
        if not other_sparse_list:
            return self.sparse_list
        merged = {}

        for encoded in self.sparse_list + list(other_sparse_list):
            idx, rank = self._decode_sparse_item(encoded)
            # only keep max rank for each index
            if idx not in merged or rank > merged[idx]:
                merged[idx] = rank

        # Re-encode: (idx << 6) | rank
        return [(idx << 6) | rank for idx, rank in merged.items()]
    
    def _decode_sparse_item(self, encoded):
        idx = encoded >> 6
        rank = encoded & 0x3F
        return idx, rank
    
    def proportion_not_k_anonymous(self, k):
        if self.mode == 'dense':
            num_maxes = []
            for idx, rank_list in enumerate(self.saved_ranks_for_k_anon_dense):
                if not rank_list:
                    continue # only process nonempty buckets
                max_value = self.registers[idx]
                print(max_value, rank_list)
                num_max = rank_list.count(max_value)
                num_maxes.append(num_max)
            num_buckets_less_than_k = sum([1 for num_max in num_maxes if 0 < num_max and num_max < k])
            return num_buckets_less_than_k/self.m
        else: # sparse
            self._merge_tmp_set()
            num_maxes = []
            for idx, hashed_list in enumerate(self.saved_data_for_k_anon_sparse):
                if not hashed_list:
                    continue # only process used indices
                max_value = [self._decode_sparse_item(item)[1] for item in self.sparse_list if self._decode_sparse_item(item)[0]==idx][0]
                hashed_ranks = [self._decode_sparse_item(pair[0])[1] for pair in hashed_list]
                num_max = hashed_ranks.count(max_value)
                # print(hashed_list)
                # print(max_value, hashed_ranks)
                num_maxes.append(num_max)
            num_buckets_less_than_k = sum([1 for num_max in num_maxes if 0 < num_max and num_max < k])
            print(num_buckets_less_than_k)
            return num_buckets_less_than_k/self.m_prime
            
if __name__ == '__main__':
    test = HyperLogLogPlusPlus(2,2)
    test.add('hi')
    print(test.mode)
    print(test.tmp_set)
    print(test.sparse_list)
    print(test.saved_data_for_k_anon_sparse)
    test.add('ha')
    print(test.mode)
    print(test.tmp_set)
    print(test.sparse_list)
    print(test.saved_data_for_k_anon_sparse)
    print(test.proportion_not_k_anonymous(2))
