import csv
import random
import numpy as np
import math
from sklearn.metrics import classification_report


def gethashtrain(dataset, trainingSet):
    newhashtrainvalues = []
    Kdim = int(len(dataset[0]))
    k1 = int(len(dataset[0]))
    k2 = int(2 ** int(math.log2(k1)))
    # Ddim = int(k2/8)
    Ddim = int(min(k2 / 2, 256))
    print('D',Ddim)
    numhashtable = 10
    numrowstrain = int(len(trainingSet))
    Randmatrix = np.zeros((numhashtable, Kdim, Ddim))
    reducedarray = np.zeros((numhashtable, numrowstrain, Ddim))
    for n in range(numhashtable):
        Randmatrix[n] = np.random.normal(0, 1, (Kdim, Ddim))
        reducedarray[n] = np.matmul(trainingSet, Randmatrix[n])
        hashvalue = (np.dot(trainingSet, Randmatrix[n]) > 0).astype(int).astype(str)
        newhashtrainvalues.append(hashvalue)
    return reducedarray, newhashtrainvalues, Randmatrix


def gethamming(str1, str2):
    sum = 0
    for char1, char2 in zip(str1, str2):
        if (char1 != char2):
            sum += 1
    return sum


def getclasslabel(labeltest, labeltrain, findind):
    labelset = []
    labelset.append(labeltrain[0][0])
    index2 = 0
    for i in range(len(labeltrain)):
        if labeltrain[i][0] in labelset:
            continue
        else:
            index2 += 1
            labelset.append(labeltrain[i][0])
            # print(labeltrain[i][0])

    index1 = 0
    count = np.zeros((len(labelset), 1), dtype=int)
    for j in range(len(labelset)):
        for i in range(len(findind)):
            if (findind[i] == labelset[j]):
                count[index1] += 1
        index1 += 1
    # print(count)
    if (len(count) == 4):
        d1 = max(count[0],count[1],count[2],count[3])
    else:
        d1 = max(count[0],count[1],count[2])
    for i in range(len(count)):
        if (d1 == count[i]):
            # print('class obtained is:', format(labelset[i]))
            classis = labelset[i]
    return classis


def getclassified(dataset, Randmatrix, newhashtrainvalues, trainingSet, testSet, labeltrain1, labeltrain, labeltest):
    k1 = int(len(dataset[0]))
    k2 = int(2 ** int(math.log2(k1)))
    # Ddim = int(k2/8)
    Ddim = int(min(k2/8, 256))
    numhashtable = 10
    numrowstrain = int(len(trainingSet))
    numrowstest = int(len(testSet))
    hammingdisthash = np.zeros((numhashtable,numrowstrain))
    findind = np.zeros((numhashtable))
    hashvaluetest = np.zeros(((Ddim)))
    classlabelfound = np.zeros((numrowstest))
    accuracycount = 0
    totalcount = 0
    tocalf1 = []
    for ntest in range(numrowstest):
        hashvaluetest = []
        for n in range(numhashtable):
            hashvaluetest = (np.dot(testSet[ntest], Randmatrix[n]) > 0).astype(int).astype(str)
            for ntrain in range(len(newhashtrainvalues[0])):
                hammingdisthash[n,ntrain] = gethamming(newhashtrainvalues[n][ntrain],hashvaluetest)
            findind[n] = labeltrain1[np.argmin(hammingdisthash[n])]
        classifiedas = getclasslabel(labeltest, labeltrain, findind)
        tocalf1.append(classifiedas)
        totalcount += 1
        if (classifiedas == labeltest[ntest][0]):
            accuracycount += 1

    return findind, classlabelfound, accuracycount, totalcount, tocalf1


def lshmain(filename, labelfilename):
    accuracycount = 0
    totalcount = 0

    labelfile = open(labelfilename, 'rt')
    lablines = csv.reader(labelfile, delimiter=',')
    list1 = list(lablines)
    for x1 in range(len(list1)):
        for y1 in range(len(list1[0])):
            list1[x1][y1] = float(list1[x1][y1])
            list1.append(list1[x1])
    kl = int(len(list1) / 2)
    list1 = list(list1[0:kl])

    lines = csv.reader(open(filename, 'rt'), delimiter=' ')
    dataset = list(lines)
    for x in range(len(dataset)):
        for y in range(len(dataset[0])):
            dataset[x][y] = float(dataset[x][y])
        dataset.append(dataset[x])
        # print(type(dataset))
    k = int(len(dataset) / 2)
    dataset = list(dataset[0:k])

    for i in range(10):
        testSet = []
        trainingSet = []
        labeltest = []
        labeltrain = []
        print(i)
        i1 = (i * int(len(dataset) / 10))
        i2 = ((i + 1) * int(len(dataset) / 10))
        for j in range(len(dataset)):
            if (i1 <= j <= i2):
                testSet.append(dataset[j])
                labeltest.append(list1[j])
            else:
                trainingSet.append(dataset[j])
                labeltrain.append(list1[j])
        labeltrain1 = []
        for i in labeltrain:
            for j in i:
                labeltrain1.append(j)
        classlabelfound = np.zeros((len(labeltest)))
        reducedarray, newhashtrainvalues, Randmatrix = gethashtrain(dataset, trainingSet)
        findind, classlabelfound, accuracycount, totalcount, tocalf1 = getclassified(dataset, Randmatrix, newhashtrainvalues, trainingSet, testSet, labeltrain1, labeltrain,
                                labeltest)
        print('Classification Report: ', classification_report(labeltest, tocalf1))
        accuracy = accuracycount / totalcount
        print('Accuracy: ',accuracy)

'''
filename = 'twitter.csv'
labelfilename = 'twitter_label.csv'
lshmain(filename, labelfilename)
'''