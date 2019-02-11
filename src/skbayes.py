from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import csv
import random
import operator

def getlist1(filename, labelfilename):
    lines = csv.reader(open(filename))
    labelfile = open(labelfilename,'rt')
    lablines = csv.reader(labelfile,delimiter=',')
    list1=list(lablines)
    for x1 in range(len(list1)):
        for y1 in range(len(list1[0])):
            list1[x1][y1] = float(list1[x1][y1])
            list1.append(list1[x1])
    kl = int(len(list1)/2)
    list1 = list(list1[0:kl])
    return list1


def loadDataset(filename):
    lines = csv.reader(open(filename,'rt'),delimiter=' ')
    dataset = list(lines)
    for x in range(len(dataset)):
        for y in range(len(dataset[0])):
            dataset[x][y] = float(dataset[x][y])
        dataset.append(dataset[x])
            # print(type(dataset))
    k = int(len(dataset)/2)
    dataset = list(dataset[0:k])
    return dataset


def kfoldcrossvalidation(dataset, list1):
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
        clf = GaussianNB()
        clf.fit(trainingSet, labeltrain)
        print('Accuracy Score: ', accuracy_score(labeltest,clf.predict(testSet)))
        print('Classification Report: ', classification_report(labeltest, clf.predict(testSet)))


def sknbmain(filename, labelfilename):
    list1 = getlist1(filename, labelfilename)
    dataset = loadDataset(filename)
    kfoldcrossvalidation(dataset, list1)
'''
filename = 'pubmed.csv'
labelfilename = 'pubmed_label.csv'
# print('pubmedoriginal')
sknbmain(filename, labelfilename)
'''