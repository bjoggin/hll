import math
import hashlib
import numpy as np
import xxhash
import copy
from collections import defaultdict

class HyperLogLogPlusPlus:
    def __init__(self, p=14, p_prime=25, regularHLL = False):
        self.mode = 'sparse'
        # for dense representation
        self.p = p
        self.m = 2 ** p
        self.alpha = self._get_alpha(self.m)
        self.registers = np.zeros(self.m, dtype=int) # buckets
        self.saved_ranks_for_k_anon_dense = [[] for _ in range(self.m)] # for k anonymity

        # for sparse representation
        self.p_prime = p_prime # sparse encoding uses up to 2^p'
        self.m_prime = 2 ** self.p_prime
        self.sparse_list = []
        self.sparse_threshold = 1600  # threshold for switching to dense mode (temporary)
        self.sparse_alpha = self._get_alpha(self.m_prime)
        self.saved_data_for_k_anon_sparse = [[] for _ in range(self.m_prime)] # for k anonymity and switching to dense

        self.regularHLL = regularHLL
        if self.regularHLL:
            self.mode = 'dense'

    def _get_alpha(self, m):
        # according to Google's paper
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)

    def _hash(self, value):
        # h = hashlib.(value.encode('utf-8')).hexdigest()
        h = xxhash.xxh64(value.encode('utf-8')).hexdigest()
        return int(h, 16)

    def add(self, value):
        x = self._hash(value)
        if self.mode == 'sparse':
            k = self._encode(x, self.p_prime) # idx and number of leading zero concatenated into 1 number
            self.sparse_list.append(k)
            idx, _ = self._decode(k)
            self.saved_data_for_k_anon_sparse[idx].append((k,x)) # for this bucket, add (encoded, hash)

            if self.exceedingSparse():
                self._convert_sparse_to_dense()
        else:
            self._update_registers(x)

    def _encode(self, x, p):
        idx = x >> (64 - p) # use first p number of bits to get bucket index
        w = x & ((1 << (64 - p)) - 1)
        rank = self._rank(w, 64 - p) # find position of first leading 1 
        return (idx << 6) | rank # concatenate, leaving 6 bits for rank (max has 2^6 leading 0s)
    
    def _rank(self, bits, max_bits):
        return max_bits - bits.bit_length() + 1
    
    def _decode(self, encoded):
        idx = encoded >> 6
        rank = encoded & 0x3F
        return idx, rank

    def merge_sparse_lists(self, other_sparse_list):
        if not other_sparse_list:
            return self.sparse_list

        merged = {}
        for encoded in self.sparse_list + list(other_sparse_list):
            idx, rank = self._decode(encoded)
            # only keep max rank for each index
            if idx not in merged or rank > merged[idx]:
                merged[idx] = rank

        # Re-encode: (idx << 6) | rank
        return [(idx << 6) | rank for idx, rank in merged.items()]
    
    def exceedingSparse(self):
        return len(self.sparse_list) > self.sparse_threshold # for testing purposes
        # return len(self.sparse_list) > self.m * 6: # official rule from the paper
    
    def _convert_sparse_to_dense(self):
        print("converting to dense")
        for item in self.saved_data_for_k_anon_sparse: # in each bucket, has (encoded, hash)
            for pair in item:
                hashed = pair[1]
                self._update_registers(hashed)
        self.saved_data_for_k_anon_sparse = []
        self.sparse_list = []
        self.mode = 'dense'

    def _update_registers(self, x):
        encoded = self._encode(x, self.p)
        idx, rank = self._decode(encoded)
        self.registers[idx] = max(self.registers[idx], rank) # update max in bucket
        self.saved_ranks_for_k_anon_dense[idx].append(rank)

    def estimate(self):
        m = self.m if self.mode == 'dense' else self.m_prime
        if self.mode == 'sparse':
            occupied = set([self._decode(encoded)[0] for encoded in self.sparse_list]) # indices of all occupied buckets
            V = m - len(occupied) # empty buckets
            if V != 0:
                return self._linear_counting(m, V)
            else: # V = 0 will give division by 0 error in linear countiing 
                print("sparse fallback")
                buckets = np.zeros(m, dtype=int)
                for encoded in self.sparse_list:
                    idx, rank = self._decode(encoded)
                    buckets[idx] = max(buckets[idx], rank)
                sparse_buckets = buckets
        registers = self.registers if self.mode == 'dense' else sparse_buckets

        Z = 1.0 / np.sum(2.0 ** -registers)
        if self.regularHLL: # no bias correcting
            return self.alpha * m * m * Z 
        E = self.alpha * m * m * Z if self.mode == 'dense' else self.sparse_alpha * m * m * Z 

        # Bias correction and thresholds
        if E <= 2.5 * m: # small cardinality
            print("small cardinality")
            V = np.count_nonzero(registers == 0)
            print(m, V, len(registers))
            if V != 0:
                return self._linear_counting(m, V)
        elif E > (1/30) * (2 ** 32): # extremely large cardinality
            print("extremely large cardinality")
            return -(2 ** 32) * math.log(1 - (E / 2 ** 32))
        print("default")
        return E # default
    
    def _linear_counting(self, m, V):
        # print("in linear count: m=", m, ", V=", V)
        return m * math.log(m / V) # V is number of zero-valued buckets

    def aggregate(self, other):
        if not isinstance(other, HyperLogLogPlusPlus):
            raise TypeError("Can only merge with another HLL++")
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different precision")
        if self.mode == 'sparse' and other.mode == 'sparse':
            self.sparse_list = self.merge_sparse_lists(other.sparse_list)
            for idx, hashed_list in enumerate(other.saved_data_for_k_anon_sparse):
                self.saved_data_for_k_anon_sparse[idx].extend(hashed_list)

        if self.mode == 'dense' or self.exceedingSparse() or other.mode == 'dense':
            print("exceeded or dense")
            self._convert_sparse_to_dense() # does nothing if already dense
            other_copy = copy.deepcopy(other)
            other_copy._convert_sparse_to_dense()
            self.registers = np.maximum(self.registers.copy(), other_copy.registers.copy()) # bucket-wise maximums
            for idx, ranks in enumerate(other_copy.saved_ranks_for_k_anon_dense):
                self.saved_ranks_for_k_anon_dense[idx].extend(ranks)
    
    def proportion_not_k_anonymous(self, k):
        if self.mode == 'dense':
            num_maxes = []
            for idx, rank_list in enumerate(self.saved_ranks_for_k_anon_dense):
                if not rank_list:
                    continue # only process nonempty buckets
                max_value = self.registers[idx]
                num_max = rank_list.count(max_value) # count how many people had the same max rank in this bucket
                num_maxes.append(num_max)
            num_buckets_less_than_k = sum([1 for num_max in num_maxes if 0 < num_max and num_max < k])
            # print(num_buckets_less_than_k)
            return num_buckets_less_than_k/self.m
        else: # sparse
            num_maxes = []
            for idx, hashed_list in enumerate(self.saved_data_for_k_anon_sparse):
                if not hashed_list:
                    continue # only process used indices
                max_value = max([self._decode(encoded)[1] for encoded in self.sparse_list if self._decode(encoded)[0]==idx])
                hashed_ranks = [self._decode(pair[0])[1] for pair in hashed_list] # [(encoded, hash)]
                num_max = hashed_ranks.count(max_value)
                num_maxes.append(num_max)
            num_buckets_less_than_k = sum([1 for num_max in num_maxes if 0 < num_max and num_max < k])
            # print(num_buckets_less_than_k)
            return num_buckets_less_than_k/self.m_prime
            
if __name__ == '__main__':
    test = HyperLogLogPlusPlus(2,2)
    test.add('hi')
    print(test.mode)
    print(test.sparse_list)
    print(test.saved_data_for_k_anon_sparse)
    test.add('ha')
    print(test.mode)
    print(test.sparse_list)
    print(test.saved_data_for_k_anon_sparse)
    print(test.proportion_not_k_anonymous(2))
