import os
import shutil

source_folder = 'C:/Users/raksh/train_data'
holdout_folder = 'C:/Users/raksh/holdout_data'
os.makedirs(holdout_folder, exist_ok=True)

all_files = os.listdir(source_folder)

basenames = {os.path.splitext(f)[0] for f in all_files if f.endswith(('.hea', '.dat'))}

# Select a fraction (20%) of records
import random
holdout_basenames = random.sample(list(basenames), int(len(basenames)*0.2))

for base in holdout_basenames:
    for ext in ['.hea', '.dat']:
        src_file = os.path.join(source_folder, base + ext)
        dst_file = os.path.join(holdout_folder, base + ext)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")
        else:
            print(f"Warning: {src_file} does not exist!")
