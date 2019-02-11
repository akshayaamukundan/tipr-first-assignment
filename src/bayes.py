# Implement Bayes Classifier here!
import csv
import math
import random
import numpy as np
from sklearn.metrics import classification_report

# NB
def getlist1(labelfilename):
    labelfile = open(labelfilename,'rt')
    lablines = csv.reader(labelfile,delimiter=',')
    list1=list(lablines)
    for x1 in range(len(list1)):
        for y1 in range(len(list1[0])):
            list1[x1][y1] = float(list1[x1][y1])
            list1.append(list1[x1])
    kl = int(len(list1) / 2)
    list1 = list(list1[0:kl])
    return list1


def loadDataset(filename):
    lines = csv.reader(open(filename,'rt'),delimiter=' ')
    dataset = list(lines)
    # print(dataset)
    # print(dataset[0][0])
    # print(type(dataset))
    for x in range(len(dataset)):
        for y in range(len(dataset[0])):
            dataset[x][y] = float(dataset[x][y])
        dataset.append(dataset[x])
        # print(type(dataset))
    k = int(len(dataset) / 2)
    dataset = list(dataset[0:k])
    # print('labl: ', labeltrain)
    # print('testlabl', labeltest)
    return dataset


def getlabelset(labeltrain):
    labelset = []
    labelset.append(labeltrain[0][0])
    index = 0
    for i in range(len(labeltrain)):
        if labeltrain[i][0] in labelset:
            continue
        else:
            index += 1
            # print(index)
            labelset.append(labeltrain[i][0])
            # print(labeltrain[i][0])

    # print(labelset)
    # print(labelset[0])
    return labelset


def getmean(classname):
    mean = np.zeros((1, len(classname[0])))
    for i in range(len(classname[0])):
        sum = 0
        for j in range(len(classname)):
            sum += classname[j][i]
        mean[0][i] = sum / len(classname)
    return mean


def getvariance(classname):
    meani = getmean(classname)
    var1 = np.zeros((1, len(classname[0])))
    for i in range(len(classname[0])):
        sum = 0
        for j in range(len(classname)):
            sum += math.pow((classname[j][i] - meani[0][i]),2)
        var1[0][i] = (sum / len(classname)) + 0.001
    return var1


def prob(x, classname):
    mean1 = getmean(classname)
    var1 = getvariance(classname)
    prob1 = []
    prob2 = []
    prod = 1
    a = 0
    for i in range(len(x)):
        a = (1 / math.sqrt(2 * math.pi * var1[0][i]))
        prob2.append(a * math.exp(-(math.pow((x[i] - mean1[0][i]), 2) / (2 * var1[0][i]))))
        # prob2.append(prob1[i] * (1 / math.sqrt(2 * math.pi * var1[0][i])))
        # print(prob1[i])
        # print(prob2[i])
        prod *= prob2[i]
    return prob2,prod


def toclassify(trainingSet, testSet, labeltrain, labeltest, accuracycount, totalcount):
    # print(type(testSet[1]))
    labelset = getlabelset(labeltrain)
    newtrainset = []
    index1 = 0
    count = np.zeros((len(labelset), 1), dtype=int)
    for j in range(len(labelset)):
        for i in range(len(labeltrain)):
            if (labeltrain[i][0] == labelset[j]):
                newtrainset.append(trainingSet[i])
                count[index1] += 1
        index1 += 1
    # print(count)
    # print(len(newtrainset))
    countsum = 0
    for i in range(len(count)):
        countsum += count[i]
    pca = count[0] / countsum
    pcb = count[1] / countsum
    pcc = count[2] / countsum
    if (count[0] + count[1] + count[2] != len(newtrainset)):
        pcd = count[3] / countsum
    # print(pca)

    classa = []
    classb = []
    classc = []
    classd = []

    for i in range(0, int(count[0]), 1):
        classa.append(newtrainset[i])
    for i in range(int(count[0]), int(count[0] + count[1]), 1):
        classb.append(newtrainset[i])
    for i in range(int(count[0] + count[1]), int(count[2] + count[0] + count[1]), 1):
        classc.append(newtrainset[i])
    if (count[0] + count[1] + count[2] != len(newtrainset)):
        for i in range(int(count[0] + count[1] + count[2]), int(count[3] + count[2] + count[0] + count[1]), 1):
            classd.append(newtrainset[i])

    meana = getmean(classa)
    meanb = getmean(classb)
    meanc = getmean(classc)
    if (count[0] + count[1] + count[2] != len(newtrainset)):
        meand = getmean(classd)

    vara = getvariance(classa)
    varb = getvariance(classb)
    varc = getvariance(classc)
    if (count[0] + count[1] + count[2] != len(newtrainset)):
        vard = getvariance(classd)

    tocalf1 = []
    for j in range(len(testSet)):
        proba, proda = prob(testSet[j], classa)
        probb, prodb = prob(testSet[j], classb)
        probc, prodc = prob(testSet[j], classc)
        if (count[0] + count[1] + count[2] != len(newtrainset)):
            probd, prodd = prob(testSet[j], classd)
        # print('actual class', format(labeltest[2]))
        netpa = netpb = netpc = netpd = 0

        netpa = pca * proda
        netpb = pcb * prodb
        netpc = pcc * prodc
        if (count[0] + count[1] + count[2] != len(newtrainset)):
            netpd = pcd * prodd

        c = max(netpa, netpb, netpc, netpd)
        if c == netpa:
            # print('Classified as:', format(labelset[0]))
            classidentifiedas = labelset[0]
        elif (c == netpb):
            # print('Classified as:', format(labelset[1]))
            classidentifiedas = labelset[1]
        elif (c == netpc):
            # print('Classified as:', format(labelset[2]))
            classidentifiedas = labelset[2]
        else:
            # print('Classified as:', format(labelset[3]))
            classidentifiedas = labelset[3]
        tocalf1.append(classidentifiedas)
        totalcount += 1
        if (classidentifiedas == labeltest[j][0]):
            accuracycount += 1
    print('Classification Report: ', classification_report(labeltest, tocalf1))
    return accuracycount, totalcount


def nbmain(filename, labelfilename):
    accuracycount = 0
    totalcount = 0
    list1 = getlist1(labelfilename)
    # print(list1)
    trainingSet = []
    testSet = []
    labeltrain = []
    labeltest = []
    dataset = loadDataset(filename)
    for i in range(10):
        testSet = []
        trainingSet = []
        labeltest = []
        labeltrain = []
        print(i)
        i1 = (i * int(len(dataset) / 10))
        i2 = ((i + 1) * int(len(dataset) / 10))
        # print(i1, i2)
        for j in range(len(dataset)):
            # if (j < int(len(dataset) / 10)):
            if (i1 <= j <= i2):
                testSet.append(dataset[j])
                labeltest.append(list1[j])
            else:
                trainingSet.append(dataset[j])
                labeltrain.append(list1[j])
        # print('test', testSet)
        # print('train', trainingSet)
        accuracycount, totalcount = toclassify(trainingSet,testSet,labeltrain, labeltest, accuracycount, totalcount)
        accuracy = accuracycount / totalcount
        print(accuracy)
'''
filename = 'twitter._2.csv'
labelfilename = 'twitter_label.csv'
nbmain(filename, labelfilename)
'''