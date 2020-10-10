import pickle
import copy
import math

TRAINING_FILE = "datasets/ionosphere.arff"
alpha = 1


with open(TRAINING_FILE, 'r') as fp:
    training_lines = fp.read().strip().split('\n')
X = list()
Y = list()
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
        Y.append(0)
    else:
        Y.append(1)

def h(theta, Xi):
    n = len(theta)
    h = 0
    for j in range(n):
        h += theta[j] * Xi[j]
    return pow(1 + math.exp(-h), -1)

def l(X, Y, theta):
    lvar = 0
    for i in range(m):
        lvar += Y[i] * math.log(h(theta, X[i])) + (1 - Y[i]) * math.log(1 - h(theta, X[i]))
    return lvar


def deriv(X, Y, theta, j):
    n = len(theta)
    assert j < n and j >= 0
    m = len(Y)
    delta = 0
    for i in range(m):
        delta += X[i][j] * (Y[i] - h(theta, X[i]))
    return delta / m

def update(X, Y, theta, alpha):
    n = len(theta)
    theta2 = copy.deepcopy(theta)
    for j in range(n):
        theta2[j] += alpha * deriv(X, Y, theta, j)
    return theta2

theta = list()
for i in range(n):
    theta.append(0)

test_index = 196

while True:
    theta2 = copy.deepcopy(theta)
    theta = update(X, Y, theta, alpha)
    diff = l(X, Y, theta) - l(X, Y, theta2)
    print l(X, Y, theta), diff, h(X[test_index], theta), Y[test_index]
    if diff <= 0.0001:
        break


print "pred: ", h(theta, X[test_index]), "expected: ", Y[test_index]
with open('log_theta', 'wb') as fp:
    pickle.dump(theta, fp)
