# Intrusion Detector
This repository provides an implementation of an anomaly classifier which detects network intrusions to protect a computer network 
from unauthorized users, including perhaps insiders. The intrusion detector learning task is to build a predictive model capable of 
distinguishing between "bad" connections, called intrusions or attacks, and "good" normal connections. 
The implementation utilizes a univariate Gaussian distribution model on the KDD Cup 1999 dataset for training and testing purposes.

# Dependencies
Package - Appropriate Python package

Frameworks - Numpy, Pandas, Scikit-learn

# Getting started
Download the essential files using the following link:

https://drive.google.com/drive/folders/1it90OQvxliYkbfPvxv2Ens1HffSyuPIe?usp=sharing

Place these files in the root folder of the repository.

# Training and testing the model
Run 'detector.py' to train and test the model.

# Result
Following files are generated in the root folder upon completion of execution.

y_true.csv - stores true labels of the processes contained in test set

y_pred.csv - stores predicted labels of the processes contained in test set

# Interpretation
y=0: Not an anomaly

y=1: Anomaly

# Performance analysis
I have placed my results in 'result'(result/) folder for reference.

No. of training examples used: 494021

No. of testing examples used: 311028

F1 Score: 0.9493410565461243
