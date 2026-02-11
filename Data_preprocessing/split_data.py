import os
import shutil
from tqdm import tqdm


source_root = r"D:\BAPS\Projects\MBC\Data_Ex\BCMID"
output_root = r"D:\BAPS\Projects\MBC\Discrete_Dataset"

mammo_target = os.path.join(output_root, "Mammogram_Dataset")
us_target = os.path.join(output_root, "Ultrasound_Dataset")

os.makedirs(mammo_target, exist_ok=True)
os.makedirs(us_target, exist_ok=True)

patients = [p for p in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, p))]

for patient in tqdm(patients):
    patient_path = os.path.join(source_root, patient)
    
    # Process Mammograms
    src_mammo = os.path.join(patient_path, "Mammogram")
    if os.path.exists(src_mammo):
        dst_mammo_folder = os.path.join(mammo_target, patient)
        os.makedirs(dst_mammo_folder, exist_ok=True)
        for img in os.listdir(src_mammo):
            shutil.copy2(os.path.join(src_mammo, img), os.path.join(dst_mammo_folder, img))
            
    # Process Ultrasound
    src_us = os.path.join(patient_path, "Ultrasound")
    if os.path.exists(src_us):
        dst_us_folder = os.path.join(us_target, patient)
        os.makedirs(dst_us_folder, exist_ok=True)
        for img in os.listdir(src_us):
            shutil.copy2(os.path.join(src_us, img), os.path.join(dst_us_folder, img))
