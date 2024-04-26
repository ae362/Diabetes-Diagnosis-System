# Diabetes-Diagnosis-System

## Project Overview
This repository contains the code for a Diabetes Diagnosis System. It leverages an Artificial Neural Network (ANN) for predicting diabetes based on various health indicators. Additionally, a Prolog-based expert system is used to deduce diagnoses logically, considering symptoms and clinical test results.

## Repository Structure
Diabetes Dataset/: This directory should contain the dataset used for training and testing the ANN model. Due to privacy or size concerns, this is often not uploaded to GitHub but mentioned in the README.md for users to obtain and place in this directory.
logs/: This folder is intended for log files generated during model training, likely from TensorBoard or similar tools. It's common practice to exclude this from version control if the logs are voluminous.
best_model.keras: This is the best-performing model saved by the ModelCheckpoint callback. It's good to have the model saved so others can use it directly without retraining.
diabetes_ANN_model.keras: This seems to be another saved state of your trained ANN model. It could be the final model state after completing all training epochs.
diagnosis/: This directory is a bit ambiguous. Typically, a .pl file would not be a directory but a file. If it is a Prolog source file, it should just be diagnosis.pl. If it's a directory, it should contain related Prolog files.
requirements/: Similar to diagnosis/, if this is intended to be the requirements.txt file, it should not be a directory but a single text file listing all the Python package dependencies.
script/: As with diagnosis/, this should not be a directory if it's meant to represent the main Python script (script.py). The Python file should contain the source code for preprocessing the data, building, training, and evaluating the ANN model.

## How to Set Up

1. Clone the repository:
```bash
git clone https://github.com/ae362/Diabetes-Diagnosis-System.git
```
2.Navigate to the project directory:
cd Diabetes-Diagnosis-System

Install the necessary dependencies:
pip install -r requirements.txt

To execute the ANN model training and evaluation:
python src/script.py

To run the Prolog-based expert system:
swipl -s src/diagnosis.pl



