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

