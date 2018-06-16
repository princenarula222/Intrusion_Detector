import pandas
import numpy as np
from sklearn.metrics import f1_score


np.set_printoptions(threshold=np.nan)

data = pandas.read_table('kddcupdata.csv', delimiter=',')                   # reading the intrusion data
data = np.array(data)
test = pandas.read_table('corrected.txt', delimiter=',')                    # reading the testing data
test = np.array(test)

ls1 = list(set(data[:, 1]))
ls2 = list(set(data[:, 2]))
ls3 = list(set(data[:, 3]))


def search(arr, val):
    for idx, value in enumerate(arr):
        if value == val:
            return idx
    return 0


def preprocess(data, anom):                                          # separating out anomalies
    t = (np.shape(data))[1]
    h = []
    nor = (np.shape(data))[0]
    for i in range(nor):
        if data[i][t-1] != 'normal.':
            anom.append(list(data[i]))
            continue
        h.append(list(data[i]))
    h = np.array(h)
    h = h[:, 0:t-1]
    anom = np.array(anom)
    anom = anom[:, 0:t-1]
    return h, anom


def buildset(data):                                             # assists in identifying useless features in normal data
    ls = []
    noc = (np.shape(data))[1]
    for i in range(noc):
        ls.append(list(set(data[:, i])))
    return ls


def process(data, ls):
    h = data
    k = 0
    noc = (np.shape(data))[1]
    for i in range(noc):
        if len(ls[i]) == 1:
            h = np.delete(h, i-k, 1)              # assists in removing useless features in normal data from entire data
            k = k+1
    data = h
    nor = (np.shape(data))[0]
    h = []
    for i in range(nor):
        data[i][1] = search(ls1, data[i][1])                                      # processing raw data
        data[i][2] = search(ls2, data[i][2])                                      # processing raw data
        data[i][3] = search(ls3, data[i][3])                                      # processing raw data
        h.append((list(data[i])))
    h = np.array(h).astype(float)
    return h


anom = []
data, anom = preprocess(data, anom)
ls = buildset(data)
data = process(data, ls)
anom = process(anom, ls)

m = np.shape(data)[0]

mean = np.sum(data, axis=0)                                  # calculating mean of every feature using vectorization
mean = mean/m


data = np.subtract(data, mean)

data = np.square(data)
var = np.sum(data, axis=0)                                   # calculating variance of every feature using vectorization
var = var/m


def gauss(i, mean, var):                                                              # gaussian pdf
    p = np.sqrt(2*np.pi*var)                                                          # vectorization
    p = 1/p
    p = np.multiply(p, np.exp(-np.matrix((np.asarray(i-mean)**2))/(2*var)))           # vectorization
    p = np.prod(p, axis=1)
    return p


r = np.shape(anom)[0]
c = np.shape(anom)[1]
p = gauss(anom, mean, var)
epsi = np.amax(p)                                                        # setting the anomaly limit value

testpos = []
testneg, testpos = preprocess(test, testpos)
testneg = process(testneg, ls)
testpos = process(testpos, ls)
test1 = np.ones((1, np.shape(testpos)[0]))
test2 = np.zeros((1, np.shape(testneg)[0]))
test = np.hstack((test1, test2))

testing = np.vstack((testpos, testneg))
r = np.shape(testing)[0]
c = np.shape(testing)[1]

p = gauss(testing, mean, var)
result = (p <= epsi).astype(float)                                   # predicting anomalies in testing data


result = np.asarray(result).reshape(-1)
test = np.asarray(test).reshape(-1)
print(result)
print(test)
df = pandas.DataFrame(test)
df.to_csv("y_true.csv")                                              # storing processed testing data
df = pandas.DataFrame(result)
df.to_csv("y_pred.csv")                                              # storing the predicted results
f1score = f1_score(test, result)

print(f1score)
