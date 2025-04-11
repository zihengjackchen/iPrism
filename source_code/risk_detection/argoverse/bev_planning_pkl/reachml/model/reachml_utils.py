"""
MIT License

Copyright (c) 2022 Shengkun Cui, Saurabh Jha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


def calculate_metrics(prediction, labels):
    flatten_probability = prediction
    flatten_prediction = np.zeros_like(flatten_probability)
    flatten_labels = labels
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    flatten_prediction[flatten_probability >= 0.5] = 1
    for i in range(0, len(flatten_prediction)):
        if flatten_labels[i] == flatten_prediction[i] and flatten_labels[i] == 1:
            TP += 1
        if flatten_labels[i] == 0 and flatten_prediction[i] == 1:
            FP += 1
        if flatten_labels[i] == flatten_prediction[i] and flatten_labels[i] == 0:
            TN += 1
        if flatten_labels[i] == 1 and flatten_prediction[i] == 0:
            FN += 1
    accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    precision = TP / (TP + FP + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    return accuracy, precision, recall, f1_score
