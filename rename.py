import os
import shutil

source_folder = 'C:/Users/raksh/training_data/ptbxl_output'
destination_folder = 'C:/Users/raksh/train_data'

os.makedirs(destination_folder, exist_ok=True)
for subfolder in os.listdir(source_folder):
    subfolder_path = os.path.join(source_folder, subfolder)
    if os.path.isdir(subfolder_path) and subfolder.isdigit():
        folder_number = int(subfolder)
        if 6000 <= folder_number <= 21000:
        
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                
                if os.path.isfile(file_path):
                    shutil.copy(file_path, destination_folder)
                    print(f"Copied {file_path} to {destination_folder}")

print("Done.")









