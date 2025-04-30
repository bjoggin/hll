import random
import string
from hllpp import *

def generate_patients_with_duplicates(n_unique, m_total, id_length=16):
    if n_unique > m_total:
        raise ValueError("Number of unique individuals cannot exceed total patients.")
    
    # Step 1: Generate n_unique truly unique IDs
    unique_patients = []
    seen = set()
    while len(unique_patients) < n_unique:
        patient_id = ''.join(random.choices(string.ascii_letters + string.digits, k=id_length))
        if patient_id not in seen:
            unique_patients.append(patient_id)
            seen.add(patient_id)
    
    # Step 2: Sample with replacement to generate m_total patients
    patients = random.choices(unique_patients, k=m_total-n_unique)
    
    patients.extend(unique_patients)
    return patients

def get_hospital_hlls(patients_total, unique_patients_total, n_hospitals, p: int, p_prime: int):
    patients_all = generate_patients_with_duplicates(unique_patients_total, patients_total, 10)
    random.shuffle(patients_all)

    hllpps = []
    for i in range(n_hospitals):
        number_per_hospital = int(patients_total/n_hospitals)
        start = i*number_per_hospital
        end = i*number_per_hospital + number_per_hospital
        patients = patients_all[start:end]
        hllpp = HyperLogLogPlusPlus(p, p_prime)
        for patient in patients:
            hllpp.add(patient)
        
        hllpps.append(hllpp)
    return hllpps