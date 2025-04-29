import numpy as np
import time

def simulation1(A,m,r,ntrial,k):
    """
    :param A: number of all patients
    :param m: number of buckets
    :param r: ratio of number of patients satisfying certain criteria to number of all patients
    :param ntrial: number of trails
    :param k: K in K-anonymity
    :return: Expectation by applying approximation1
    """
    B = int(A*r)  # number of patients satisfying certain criteria
    result = 0 # total number of collision in all trials
    b_temp = np.random.choice(np.arange(0, A, 1), B, replace=False) # randomly generate patients satisfying certain criteria
    partition_total = np.random.multinomial(A, [1 / m] * m, size=ntrial)
    all_estimates = []
    # run all trials
    for n in range(0, ntrial):
        partition = partition_total[n]
        e = [] # set of collision
        current_index = 0
        bucket_estimates = []
        for i in range(0, m):
            x = np.floor(-np.log2(1 - np.random.rand(partition[i])))
            y_size = np.size(np.intersect1d(b_temp,np.arange(current_index,current_index + partition[i],1)))
            y = x[0:y_size]
            current_index = current_index + partition[i]
            if np.size(y) == 0 or np.size(x) == 0:
                e.append(0)
            else:
                index = np.where(x == np.max(y))
                e.append(np.size(index))
            bucket_estimates.append(2**np.max(y))
        all_estimates.append(np.sum(bucket_estimates))
        e = np.array(e)
        result = result + np.size(np.where((0 < e) & (e < k)))

    # mean value of n trials
    return result/ntrial, all_estimates

if __name__ == '__main__':

    A = 10**4
    r = 0.1
    B = round(A*r)
    m = 100
    p = 1/m
    k = 10
    ntrial = 100
    print(simulation1(A,m,r,ntrial,k))