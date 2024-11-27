import scipy.io

file_path = 'seed_pre/drive-download-20241126T210900Z-001/1_20131027.mat'
data = scipy.io.loadmat(file_path)

labels_data = scipy.io.loadmat('seed_pre/drive-download-20241126T210900Z-001/label.mat')

# Print all the keys in the loaded .mat file
print(labels_data.keys())

eeg_data = [data[f'djc_eeg{i}'] for i in range(1, 16)]
# Assuming 'labels_data' is the loaded dictionary from label.mat
labels = labels_data['label']

