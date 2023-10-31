#!/usr/bin/env python

from helper_code import *
import shutil
data_folder = "/media/jacobo/NewDrive/physionet.org/files/i-care/2.0/training"
save_folder = "/media/jacobo/NewDrive/physionet.org/files/i-care/2.0/validation"
patient_ids  = os.listdir(data_folder) #"0286"

hospital_A = list() 
CPC_1 = 0
for patient_id in patient_ids:
    patient_metadata = load_challenge_data(data_folder, patient_id)
    hospital = get_hospital(patient_metadata)
    if hospital == "F":
        hospital_A.append(patient_id)
        # Use shutil.move() to cut and paste the folder
        source_folder = os.path.join(data_folder, patient_id)
        destination_folder = os.path.join(save_folder, patient_id)
        shutil.move(source_folder, destination_folder)
    
        CPC = get_outcome(patient_metadata)
        if CPC == 1:
            CPC_1 +=1

print("Done ....")
print(len(hospital_A))
print(len(hospital_A)-CPC_1)
print(CPC_1)
