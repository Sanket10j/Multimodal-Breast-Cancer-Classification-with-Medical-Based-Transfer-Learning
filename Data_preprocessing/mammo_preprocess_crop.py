import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

import cv2
import os
import numpy as np
from tqdm import tqdm  # pip install tqdm

def crop_breast_robust(img):
    if img is None:
        return None, False
        
    # if len(img.shape) == 3:
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # else:
    #     gray = img

    # 2. Invert because background is white (255) and breast is darker
    inverted = cv2.bitwise_not(img)

    # 3. Threshold (100 is aggressive enough to ignore noise)
    _, thresh = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)

    # 4. Aggressively remove the border to kill scanner artifacts
    h, w = thresh.shape
    margin_h = int(h * 0.03) 
    margin_w = int(w * 0.03)
    
    mask_cleaned = np.zeros_like(thresh)
    # Copy only the center part of the mask
    mask_cleaned[margin_h:h-margin_h, margin_w:w-margin_w] = thresh[margin_h:h-margin_h, margin_w:w-margin_w]
    
    # 5. Find contours on the CLEANED mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img, False

    # 6. Get the largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(c)
    
    # 7. Add Padding (Your tuned values: 12% W, 11% H)
    pad_w = int(w_box * 0.12)
    pad_h = int(h_box * 0.11)
    
    x_start = max(0, x - pad_w)
    y_start = max(0, y - pad_h)
    x_end = min(w, x + w_box + pad_w)
    y_end = min(h, y + h_box + pad_h)

    cropped = img[y_start:y_end, x_start:x_end]
    return cropped, True



# 1. Set your root dataset path
root_dataset_path = r"D:\BAPS\Projects\MBC\Data - Copy\BCMID"
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

# 2. Get all patient folders
patient_folders = [f for f in os.listdir(root_dataset_path) if os.path.isdir(os.path.join(root_dataset_path, f))]

print(f"Found {len(patient_folders)} patients. Starting Batch Crop...")

# 3. Iterate with Progress Bar
for patient_id in tqdm(patient_folders):
    patient_mammo_dir = os.path.join(root_dataset_path, patient_id, "Mammogram")
    
    # Check if Mammogram folder exists for this patient
    if not os.path.exists(patient_mammo_dir):
        continue
        
    # Process every image in the folder
    for img_name in os.listdir(patient_mammo_dir):
        if img_name.lower().endswith(valid_extensions):
            full_img_path = os.path.join(patient_mammo_dir, img_name)
            
            # Read Image
            original_img = cv2.imread(full_img_path)
            
            if original_img is None:
                continue
            
            # Crop
            cropped_img, success = crop_breast_robust(original_img)
            
            # Only overwrite if cropping was actually successful (found a contour)
            if success:
                # Overwrite the original file
                cv2.imwrite(full_img_path, cropped_img)
            else:
                # Optional: Log failures if you want to inspect them later
                print(f"Skipped (No Contour): {patient_id}/{img_name}")

print("\ndone")

# def crop_breast_robust(img):
#     if img is None:
#         return None, False
        
#     # 1. Convert to grayscale if needed
#     if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img

#     # 2. Invert because background is white (255) and breast is darker
#     # We want the breast to be 'white' for contour detection
#     inverted = cv2.bitwise_not(gray)

#     # 3. Apply a strong threshold to remove the 'L' and grey background noise
#     # We use a value of 50 to 80 to ensure we only get the 'thick' part of the breast
#     _, thresh = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)

#     # 4. Aggressively remove the border from the mask
#     # This 'cuts' the connection between the breast and the scanner edges
#     h, w = thresh.shape
#     # Increase border removal to 2% or 3% to be safe
#     margin_h = int(h * 0.03) 
#     margin_w = int(w * 0.03)
    
#     # Create a black frame around the thresholded mask
#     mask_cleaned = np.zeros_like(thresh)
#     mask_cleaned[margin_h:h-margin_h, margin_w:w-margin_w] = thresh[margin_h:h-margin_h, margin_w:w-margin_w]
    
#     # 5. Find contours on the CLEANED mask
#     contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if not contours:
#         return img, False

#     # 6. Get the largest contour
#     c = max(contours, key=cv2.contourArea)
#     x, y, w_box, h_box = cv2.boundingRect(c)
    
#     # 7. Add 5% padding back so we don't clip the edges of the skin
#     pad_w = int(w_box * 0.12)
#     pad_h = int(h_box * 0.11)
    
#     x_start = max(0, x - pad_w)
#     y_start = max(0, y - pad_h)
#     x_end = min(w, x + w_box + pad_w)
#     y_end = min(h, y + h_box + pad_h)

#     cropped = img[y_start:y_end, x_start:x_end]
#     return cropped, True

# # --- 2. Setup Specific Patient Path ---
# # Note: Use 'r' before the string for Windows paths
# patient_path = r"D:\BAPS\Projects\MBC\Data - Copy\BCMID\1173y15\Mammogram"

# # Get all image files (png, jpg, jpeg)
# valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
# image_files = [f for f in os.listdir(patient_path) if f.lower().endswith(valid_extensions)]

# # --- 3. Run Test on First 3 Images ---
# plt.figure(figsize=(15, 5))

# for i, img_name in enumerate(image_files[:3]): # Check first 3 images
#     full_path = os.path.join(patient_path, img_name)
#     original = cv2.imread(full_path)
    
#     if original is None:
#         continue

#     cropped, success = crop_breast_robust(original)

#     # Visualization Logic
#     plt.subplot(2, 3, i+1)
#     plt.imshow(original, cmap='gray')
#     plt.title(f"Original: {img_name}")
#     plt.axis('off')
    
#     plt.subplot(2, 3, i+4)
#     plt.imshow(cropped, cmap='gray')
#     plt.title("Cropped ROI")
#     plt.axis('off')

# plt.tight_layout()
# plt.show()
