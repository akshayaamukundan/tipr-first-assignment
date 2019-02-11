# Implement code for random projections here!
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import csv
import random
import math
import operator
import numpy as np

# Random Projections
def getlist1(filename, labelfilename):
    lines = csv.reader(open(filename))
    labelfile = open(labelfilename,'rt')
    lablines = csv.reader(labelfile,delimiter=',')
    list1=list(lablines)
    return list1


def loadDataset(filename):
    lines = csv.reader(open(filename,'rt'),delimiter=' ')
    dataset = list(lines)
    for x in range(0,len(dataset)-1,1):
        for y in range(0,len(dataset[0])-1,1):
            dataset[x][y] = float(dataset[x][y])
            dataset.append(dataset[x])
            rn = len(dataset)
            cn = len(dataset[0])
            return dataset[0:-1]


def getrn(dataset):
    rn = len(dataset)
    return rn


def getcn(dataset):
    cn = len(dataset[0])
    return cn


def getlowd(filename, dataset, cn, rn):
    tempStr = filename.split("csv")
    k1 = len(dataset[0])
    for i in range(1, int(math.log2(k1)), 1):
        Ddim = int(2**i)
        m32byd = np.zeros((cn, Ddim), dtype=float)
        m32byd = np.random.rand(cn, Ddim)
        w = np.array(dataset, dtype=np.float)
        res = np.zeros((rn, Ddim), dtype='float')
        for i in range(len(w)):
            for j in range(len(m32byd[0])):
                for k in range(len(m32byd)):
                    res[i][j] += w[i][k] * m32byd[k][j]
        fileName = tempStr[0] + '_{}.csv'.format(Ddim)
        with open(fileName, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(res)
        f.close()

def myprojmain(filename, labelfilename):
    list1 = getlist1(filename, labelfilename)
    # print(list1)
    d = loadDataset(filename)
    # print(d)
    rn1 = getrn(d)
    cn1 = getcn(d)
    # print(rn1)
    # print(cn1)
    getlowd(filename=filename, dataset=d, rn=rn1, cn=cn1)


'''
filename = 'dolphins.csv'
labelfilename = 'dolphins_label.csv'
myprojmain(filename, labelfilename)
'''
