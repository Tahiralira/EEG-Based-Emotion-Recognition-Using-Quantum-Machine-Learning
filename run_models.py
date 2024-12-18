import streamlit as st
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_def import QuantumLayer, HybridModel  # Make sure this is correct

# Helper functions
def load_data(directory):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pt')]
    all_eeg_data = []
    all_labels = []

    for file_path in file_paths:
        data = torch.load(file_path)
        all_eeg_data.append(data['eeg'])
        all_labels.append(data['label'])

    all_eeg_data = torch.stack(all_eeg_data)
    all_labels = torch.tensor(all_labels)
    return all_eeg_data, all_labels

def load_model(model_path):
    model = HybridModel()  # Ensure HybridModel is correctly defined in model_def
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def plot_eeg(data):
    # Define EEG channels and their colors (optional)
    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5']
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # Assigning specific colors to each channel
    time_points = data.shape[-1]  # Assuming data shape is compatible

    # Create a more complex signal pattern
    t = np.linspace(0, 1, time_points)
    base_signal = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz signal
    noise_scale = 0.15  # Reduce the noise scale for more realistic EEG noise

    plt.figure(figsize=(20, 4))
    for i, channel in enumerate(channels):
        # Each channel data is the base signal with added noise and a slight offset for better visibility
        eeg_channel_data = base_signal + np.random.normal(scale=noise_scale, size=time_points) + 0.2 * i
        plt.plot(t, eeg_channel_data, label=channel, color=colors[i])

    plt.title("Sample EEG Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(loc='upper right')
    plt.grid(True)
    st.pyplot(plt)

# Mapping labels to text
label_map = {0: "Positive", 1: "Neutral", 2: "Negative"}

# Visualization function with interactive elements
def visualize_predictions(eeg_data, labels, models):
    st.title('EEG-Based Emotion Recognition Predictions')
    if 'indices' not in st.session_state:
        # Initialize random indices in session state
        st.session_state.indices = torch.randperm(len(eeg_data))[:5]

    for idx in st.session_state.indices:
        st.subheader(f'Sample {idx.item()}')
        plot_eeg(eeg_data[idx])
        truth = labels[idx].item()
        st.write(f'True Label: {label_map[truth]}')  # Use the label map here

        if st.button('Run Quantum Circuit', key=f'button_{idx.item()}'):
            sample_data = eeg_data[idx].unsqueeze(0)  # Add batch dimension
            col1, col2, col3, col4, col5 = st.columns(5)
            columns = [col1, col2, col3, col4, col5]
            for i, (model, col) in enumerate(zip(models, columns)):
                with col:
                    prediction = model(sample_data).argmax().item()  # Assuming classification task
                    st.write(f'Model {i+1} Prediction: {label_map[prediction]}')  # Also map model prediction to text

# Load models and data
model_directory = './'
model_files = [f for f in os.listdir(model_directory) if f.endswith('.pth')]
models = [load_model(os.path.join(model_directory, f)) for f in model_files]

directory = r'C:\Users\tahir\OneDrive\Desktop\QC\new_seed\processed_eeg_data'
eeg_data, labels = load_data(directory)
X_train, X_test, y_train, y_test = train_test_split(eeg_data, labels, test_size=0.2, random_state=42)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from model_def import QuantumLayer, HybridModel  # Make sure this is correct

def generate_eeg_signal(time_points):
    t = np.linspace(0, 1, time_points)
    signal = 0.5 * np.sin(2 * np.pi * 5 * t) + np.random.normal(scale=0.1, size=time_points)
    return t, signal

# Function to plot EEG signal
def plot_eeg_signal(t, signal, title):
    plt.figure(figsize=(10, 2))
    plt.plot(t, signal, label="EEG Signal", color="blue")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Simulate Quantum Circuit processing
def simulate_quantum_circuit():
    time_points = 100
    t, signal = generate_eeg_signal(time_points)

    # Step 1: Show the initial EEG signal
    st.subheader("Step 1: Displaying Initial EEG Signal")
    plot_eeg_signal(t, signal, "Initial EEG Signal")
    time.sleep(1)

    # Step 2: Simulate Angle Embedding
    st.subheader("Step 2: Angle Embedding")
    angles = np.arctan(signal)  # Example: Map EEG signal to angles
    plt.figure(figsize=(10, 2))
    plt.stem(t, angles, label="Angle Embedding", linefmt='g-', markerfmt='go', basefmt=" ")
    plt.title("Angle Embedding of EEG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (radians)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
    time.sleep(1)

    # Step 3: Simulate Quantum Circuit Layers
    st.subheader("Step 3: Passing Through Quantum Circuit Layers")
    fig_placeholder = st.empty()
    for layer in range(1, 4):  # Simulate 3 quantum layers
        updated_signal = angles * (1 + layer * 0.1)  # Example transformation
        plt.figure(figsize=(10, 2))
        plt.plot(t, updated_signal, label=f"Layer {layer} Output", color="orange")
        plt.title(f"Quantum Circuit - Layer {layer}")
        plt.xlabel("Time (s)")
        plt.ylabel("Transformed Signal")
        plt.grid(True)
        plt.legend()
        fig_placeholder.pyplot(plt)
        time.sleep(1)

    # Step 4: Final Emotion Output
    st.subheader("Step 4: Final Emotion Output")
    emotion = "Positive"  # Example output
    plt.figure(figsize=(10, 2))
    plt.plot(t, np.zeros_like(t), label=f"Emotion: {emotion}", color="red")
    plt.title(f"Final Output Emotion: {emotion}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)



# Main Streamlit application
if __name__ == '__main__':
    visualize_predictions(X_train, y_train, models)
    st.title("EEG Signal to Emotion Simulation")
    
    if st.button("Start Quantum Circuit Simulation"):
        simulate_quantum_circuit()
