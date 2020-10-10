import pickle
import math
from sklearn.metrics import classification_report, confusion_matrix
import pandas
import matplotlib.pyplot as plt

TRAINING_FILE = "datasets/ionosphere.arff"

with open(TRAINING_FILE, 'r') as fp:
    training_lines = fp.read().strip().split('\n')
X = list()
true_Y = list()
li = 0
while True:
    line = training_lines[li].strip()
    if len(line) and line[0] not in ('%', '@'):
        break
    else:
        li += 1
training_lines = training_lines[li:]
m = len(training_lines)
n = len(training_lines[0].strip().split(','))
for i in range(m):
    data = training_lines[i].strip().split(',')
    X.append(list())
    X[i].append(1)
    for j in range(n - 1):
        X[i].append(float(data[j]))
    if data[n - 1] == 'b':
        true_Y.append(0)
    else:
        true_Y.append(1)

def h(theta, Xi):
    n = len(theta)
    h = 0
    for j in range(n):
        h += theta[j] * Xi[j]
    return pow(1 + math.exp(-h), -1)

with open('log_theta', 'rb') as fp:
    theta = pickle.load(fp)

pred_Y = list()
for i in range(m):
    pred = h(theta, X[i])
    pred_Y.append(int(round(pred)))

labels = [0, 1]
data_frame = pandas.DataFrame(columns=labels, index=labels)
data_conf_matrix = confusion_matrix(true_Y, pred_Y)
for x, y in zip(data_conf_matrix, labels):
    data_frame[y] = x
print("\n\n\n--------------Confusion Matrix----------------")
print(data_frame)
print("\n\n\n-------------Classification Report------------")
print(classification_report(true_Y, pred_Y))
for j in range(n):
    input_feature = []
    for i in range (m):
        input_feature.append(X[i][j])
    plt.plot(input_feature,true_Y,'r.',input_feature,pred_Y,'bx')
    plt.show()
