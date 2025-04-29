#!/usr/bin/python
import numpy as np

def simulation1_hllpp(A, m, r, ntrial, k, sparse_threshold=100):
    """
    Simulation based on HyperLogLog++ ideas
    """
    B = int(A * r)  # number of patients satisfying the criteria
    result = 0
    b_temp = np.random.choice(np.arange(0, A, 1), B, replace=False)  # query patients
    partition_total = np.random.multinomial(A, [1 / m] * m, size=ntrial)

    for n in range(ntrial):
        partition = partition_total[n]
        e = []
        current_index = 0

        for i in range(m):
            num_patients = partition[i]

            if num_patients == 0:
                e.append(0)
                continue

            # Patients assigned to this bucket
            patients_in_bucket = np.arange(current_index, current_index + num_patients)

            # Query patients in this bucket
            query_patients_in_bucket = np.intersect1d(b_temp, patients_in_bucket)

            if num_patients <= sparse_threshold:
                # === SPARSE MODE ===
                # Simulate real hashed values uniformly between 0 and 1
                hash_values = np.random.rand(num_patients)
                # For query patients
                query_indices = np.isin(patients_in_bucket, query_patients_in_bucket).nonzero()[0]
                query_hashes = hash_values[query_indices]

                if query_hashes.size == 0:
                    e.append(0)
                else:
                    # Find minimum hash value (since small hash = high leading zeros)
                    target_hash = np.min(query_hashes)
                    collisions = np.sum(hash_values <= target_hash)  # conservative collision definition
                    e.append(collisions)

            else:
                # === DENSE MODE (Normal HLL behavior) ===
                # Simulate leading zero counts
                leading_zero_counts = np.floor(-np.log2(1 - np.random.rand(num_patients)))
                query_indices = np.isin(patients_in_bucket, query_patients_in_bucket).nonzero()[0]
                query_leading_zeros = leading_zero_counts[query_indices]

                if query_leading_zeros.size == 0:
                    e.append(0)
                else:
                    target_max = np.max(query_leading_zeros)
                    collisions = np.sum(leading_zero_counts == target_max)
                    e.append(collisions)

            current_index += num_patients

        e = np.array(e)
        result += np.size(np.where((0 < e) & (e < k)))

    return result / ntrial

if __name__ == '__main__':
    A = 10**4
    r = 0.1
    m = 100
    k = 10
    ntrial = 100
    print(simulation1_hllpp(A, m, r, ntrial, k))
