from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Clean up the data first

# Fetch a dataset of digits
digits = load_digits()

# Split into test and training set, use 75% data to train the model
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42)

print(digits)

# Get the k-nearest neighbours model
model = KNeighborsClassifier(n_neighbors=3)


def showKScores():
    ks = [i for i in range(2, 10)]
    scores = []
    for k in ks:
      # Based on the plotting this is the most accurate k
        model = KNeighborsClassifier(n_neighbors=3)
        score = cross_val_score(model, x_train, y_train, cv=5)
        score.mean()
        scores.append(score.mean())
    plt.plot(scores, ks)
    plt.xlabel('accuracy')
    plt.ylabel('k')
    plt.show()
