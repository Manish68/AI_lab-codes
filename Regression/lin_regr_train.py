import pickle
import copy

TRAINING_FILE = "datasets/housing.arff"
alpha = 0.000001


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
    Y.append(float(data[n-1]))

def h(theta, Xi):
    n = len(theta)
    h = 0
    for j in range(n):
        h += theta[j] * Xi[j]
    return h

def cost(X, Y, theta):
    m = len(Y)
    J = 0
    for i in range(m):
        J += pow(h(theta, X[i]) - Y[i], 2)
    return J / (2*m)

def deriv(X, Y, theta, j):
    n = len(theta)
    assert j < n and j >= 0
    m = len(Y)
    delta = 0
    for i in range(m):
        delta += X[i][j] * (h(theta, X[i]) - Y[i])
    return delta / m

def update(X, Y, theta, alpha):
    n = len(theta)
    theta2 = copy.deepcopy(theta)
    for j in range(n):
        theta2[j] -= alpha * deriv(X, Y, theta, j)
    return theta2

theta = list()
for i in range(n):
    theta.append(0)
while True:
    theta2 = copy.deepcopy(theta)
    theta = update(X, Y, theta, alpha)
    diff = cost(X, Y, theta2) - cost(X, Y, theta)
    print cost(X, Y, theta), diff, h(X[0], theta), Y[0]
    if diff <= 0.00001:
        break
with open('lin_theta', 'wb') as fp:
    pickle.dump(theta, fp)
