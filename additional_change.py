import os

folder = 'C:/Users/raksh/train_data'  # Adjust the folder path if necessary

for file in os.listdir(folder):
    if " - Copy" in file and (file.endswith('.dat') or file.endswith('.hea')):
        file_path = os.path.join(folder, file)
        os.remove(file_path)
        print(f"Removed: {file_path}")

print("Removal of files with ' - Copy' in their name is complete.")
