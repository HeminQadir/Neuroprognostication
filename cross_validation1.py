import os
import shutil

# Define the paths
training_data = '/home/naim/downloads/TrainingData'
validation_data = '/home/naim/downloads/ValidationData'
test_data = '/home/naim/downloads/TestData'

# Create the validation and test data folders if they don't exist
if not os.path.exists(validation_data):
    os.makedirs(validation_data)
if not os.path.exists(test_data):
    os.makedirs(test_data)

# Get the list of subfolders in the training data folder
subfolders = [f.name for f in os.scandir(training_data) if f.is_dir()]

# Copy the required subfolders to the validation data folder
for folder in ['folderA', 'folderB', 'folderC', 'folderD']:
    src = os.path.join(training_data, folder)
    dst = os.path.join(validation_data, folder)
    shutil.copytree(src, dst)

# Copy the remaining subfolder to the test data folder
folder = 'folderE'
src = os.path.join(training_data, folder)
dst = os.path.join(test_data, folder)
shutil.copytree(src, dst)

print("ValidationData folders:", os.listdir(validation_data))
print("TestData folders:", os.listdir(test_data))
