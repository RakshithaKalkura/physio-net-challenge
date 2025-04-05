import os
import shutil


source_base = 'C:/Users/raksh/training_data'
target_folder = 'C:/Users/raksh/train_data'


os.makedirs(target_folder, exist_ok=True)


code15_subs = ['exams_part12', 'exams_part15']
code15_path = os.path.join(source_base, 'code15_output')


ptbxl_path = os.path.join(source_base, 'ptbxl_output')
ptbxl_subs = [f"{i:05d}" for i in range(1000, 22000, 1000)]


samitrop_path = os.path.join(source_base, 'samitrop_output')

def copy_files_recursive(source_folder, prefix):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(('.hea', '.dat')):
                src = os.path.join(root, file)
                new_name = f"{prefix}_{file}"
                dst = os.path.join(target_folder, new_name)
                shutil.copy(src, dst)


for sub in code15_subs:
    full_path = os.path.join(code15_path, sub)
    copy_files_recursive(full_path, f"code15_{sub}")


copy_files_recursive(samitrop_path, "samitrop")

for sub in ptbxl_subs:
    full_path = os.path.join(ptbxl_path, sub)
    if os.path.exists(full_path):
        copy_files_recursive(full_path, f"ptbxl_{sub}")

print(".hea and .dat files copied into 'train_data'.")
