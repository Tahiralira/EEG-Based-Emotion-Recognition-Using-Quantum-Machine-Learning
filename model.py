import scipy.io
import pennylane as qml
import os
from sklearn.model_selection import train_test_split

# file_path = 'seed_pre/drive-download-20241126T210900Z-001/1_20131027.mat'
# data = scipy.io.loadmat(file_path)

# Print all the keys in the loaded .mat file
# print(data.keys())

directory = 'seed_pre/drive-download-20241126T210900Z-001'
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mat')]
all_eeg_data = []
all_labels = []

for file in files:
    filename = os.path.basename(file)
    data = scipy.io.loadmat(file)
    
    if filename == 'label.mat':
        labels = data['label']
        all_labels = labels.flatten()  # Flatten the array if necessary
        continue

    # Identify the prefix used for EEG data in this file
    eeg_keys = [key for key in data.keys() if '_eeg1' in key]
    if not eeg_keys:
        print(f"No EEG data found in {filename}.")
        continue

    prefix = eeg_keys[0].split('_eeg1')[0]  # Get the prefix part before '_eeg1'
    
    try:
        eeg_data = [data[f'{prefix}_eeg{i}'] for i in range(1, 16)]
        all_eeg_data.extend(eeg_data)  # Assuming all files have the same structure
    except KeyError as e:
        print(f"KeyError for file {filename}: {e}")

# Optionally print some data to check if it's loaded correctly
print(f"Loaded EEG data from {len(all_eeg_data)} segments.")
print(f"Loaded labels count: {len(all_labels)}")


dev = qml.device('default.qubit', wires=2)

#the model
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(2))
    qml.templates.BasicEntanglerLayers(weights, wires=range(2))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(2)]

# Initialize weights: random
weight_shapes = {"weights": (1, 2)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# Flatten the data if necessary
eeg_data_flattened = [trial for subject in all_eeg_data for trial in subject]
labels_flattened = [label for subject_labels in all_labels for label in subject_labels]

X_train, X_test, y_train, y_test = train_test_split(
    eeg_data_flattened, labels_flattened, test_size=0.2, random_state=42
)
