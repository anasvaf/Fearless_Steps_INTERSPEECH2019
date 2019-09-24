# Two-Dimensional Convolutional Recurrent Neural Networks for Speech Activity Detection
Created by Eleftherios Fanioudakis and Anastasios Vafeiadis

## Introduction
This repository contains the code for Task 1 (Speech Activity Detection) of the Fearless Steps Challenge.
More details about the challenge can be found at <a href="http://fearlesssteps.exploreapollo.org/index.html" target="_blank">Fearless Steps</a>.

You can also check our paper that was accepted at INTERSPEECH 2019 <a href="https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1354.html" target="_blank">Two-Dimensional Convolutional Recurrent Neural Networks for Speech Activity Detection</a>

## Explanation of the Speech Activity Detection (SAD) Task
Four system output possibilities are considered:
* True Positive (TP) – system correctly identifies start-stop times of speech segments compared to the reference (manual annotation),
* True Negative (TN) – system correctly identifies start-stop times of non-speech segments compared to reference,
* False Positive (FP) – system incorrectly identifies speech in a segment where the reference identifies the segment as non-speech, and
* False Negative (FN) – system missed identification of speech in a segment where the reference identifies a segment as speech.

SAD error rates represent a measure of the amount of time that is misclassified by the systems segmentation of the test audio files. Missing, or failing to detect, actual speech is considered a more serious error than misidentifying its start and end times.

The following link explains the Decision Cost Function (DCF) metric, as well as the '.txt' output file format:
<a href="https://exploreapollo-audiodata.s3.amazonaws.com/fearless_steps_challenge_2019/v1.0/Fearless_Step_Evaluation_Plan_v1.2.pdf" target="_blank">Evaluation Plan</a>. In particular look at pages: 14-16 and 25.

## Explanation of the Python Scripts
Library Prerequisites 
* <a href="https://github.com/jameslyons/python_speech_features" target="_blank">python_speech_features</a>
* <a href="https://librosa.github.io/librosa/" target="_blank">LibROSA</a>
* Python 2.7 (The scripts can also run with Python 3.5 and above)

### extract_sad.py
This script processes the 30 min recordings for training and evaluation into 1 sec chunks (8000 samples).
We target this problem as a multi-label problem. Despite having two labels (0: non-speech and 1: speech), we will have 8000 different labels for each 1 s wav file.
The script saves a NumPy array for each 1 sec file with a corresponding NumPy array for its labels.
You can run the script as `python extract_sad.py train`, for the Train files and `python extract_sad.py test` for the Eval files.

