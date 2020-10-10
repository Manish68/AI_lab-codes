import pickle
import math
import matplotlib.pyplot as plt

TRAINING_FILE = "datasets/housing.arff"
m = 0


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
    true_Y.append(float(data[n - 1]))

def h(theta, Xi):
    n = len(theta)
    h = 0
    for j in range(n):
        h += theta[j] * Xi[j]
    return h

with open('lin_theta', 'rb') as fp:
    theta = pickle.load(fp)

s = 'Y ='
for i in range(n):
    s = s + ' (%f)*X[%d] +'%(theta[i], i)
s = s[:-1]
print s

pred_Y = list()
for i in range(m):
    pred = h(theta, X[i])
    pred_Y.append(pred)

def err_rms(pred, true):
    assert len(pred) == len(true)
    m = len(true)
    err = 0
    for i in range(m):
        err += pow(true[i] - pred[i], 2)
    err = math.sqrt(err / m)
    return err

def err_abs(pred, true):
    assert len(pred) == len(true)
    m = len(true)
    err = 0
    for i in range(m):
        err += abs(true[i] - pred[i])
    err /= m
    return err

print "absolute error: ", err_abs(pred_Y, true_Y)
print "rms error:      ", err_rms(pred_Y, true_Y)
for j in range(n):
    input_feature = []
    for i in range (m):
        input_feature.append(X[i][j])
    plt.plot(input_feature,true_Y,'r.',input_feature,pred_Y,'bx',)
    plt.show()
