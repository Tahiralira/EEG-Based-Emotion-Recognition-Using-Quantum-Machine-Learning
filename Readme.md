# EEG-Based Emotion Recognition Using Quantum Machine Learning

This repository contains the implementation of a hybrid quantum-classical framework for emotion recognition using EEG data. The project leverages quantum machine learning techniques to enhance feature extraction and classification tasks associated with EEG signals.

## Project Overview

The aim of this project is to explore the potential of Quantum Neural Networks (QNNs) in processing EEG data to recognize human emotions. This approach is expected to tap into the quantum advantages of processing information, potentially leading to more accurate and efficient emotion recognition systems.

### Features

- **Quantum Feature Extraction**: Utilizes quantum circuits as feature extractors to process EEG data.
- **Hybrid Model Architecture**: Integrates quantum layers with classical neural network models for emotion classification.
- **Dataset**: Employs publicly available EEG datasets (SEED and Kaggle) tailored to emotion recognition tasks.

## Installation

To set up a local development environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Tahiralira/EEG-Based-Emotion-Recognition-Using-Quantum-Machine-Learning.git
cd EEG-Based-Emotion-Recognition-Using-Quantum-Machine-Learning

# Install required libraries
pip install -r requirements.txt
```

## Usage
To run the emotion recognition training scripts, execute the following command
For SEED Implementation:
```bash
Ctrl + Enter on each cell of newqnn.ipynb
```
For Kaggle DEAP Implementation:
```bash
Ctrl + Enter on each cell of QC_kaggle.ipynb
```
## Testing
To test the saved .pth models that we trained or that you train from the above scripts
In the Main Project directory
```bash
pip install -r requirements.txt
```
Open terminal
```bash
Cntrl + `
```
Run this script
```bash
python run_models.py
```

## Structure

- `data/`: The Dataset of SEED is restricted under EULA guidelines from being publically shared but you may request access from https://bcmi.sjtu.edu.cn/home/seed/seed.html
- `Kaggle/`: Includes the Python scripts for the quantum and classical models implemented with Kaggle.

## Contributing

Contributions are welcome, and any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the Aheed Umer Bilal License. See `LICENSE` for more information.

## Contact

Aheed, Umer, Bilal - k214517@nu.edu.pk

Project Link: [https://github.com/Tahiralira/EEG-Based-Emotion-Recognition-Using-Quantum-Machine-Learning](https://github.com/Tahiralira/EEG-Based-Emotion-Recognition-Using-Quantum-Machine-Learning)

