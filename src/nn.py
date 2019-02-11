# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# from sklearn.naive_bayes import MultinomialNB
import numpy as np
import csv
import operator
import  math


def eucldistance(item1, item2, length):
    distance = 0
    for x in range(length):
        distance += (item1[x] - item2[x])**2
    return math.sqrt(distance)


def getneighbor(trainingSet, testSet, labeltest, list1, dataset, accuracycount, totalcount):
    tocalf1 = []
    accuracycount = 0
    totalcount = 0
    for j in range(len(testSet)):
        distance = []
        k = 5
        length = len(testSet[j])
        for i in range(len(trainingSet)):
            distance1 = eucldistance(testSet[j], trainingSet[i],len(testSet[j]))
            distance.append((trainingSet[i],distance1))
        distance.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distance[x][0])
        classifiedas = getclassified(list1, neighbors, dataset)
        tocalf1.append(classifiedas)
        totalcount += 1
        if (classifiedas == labeltest[j][0]):
            accuracycount += 1
    return neighbors, accuracycount, totalcount, tocalf1


def getclassified(list1, neighbors, dataset):
    totnum = {}
    response = []
    for x in range(len(neighbors)):
        rownum = dataset.index(neighbors[x])
        response = list1[rownum][0]
        if response in totnum:
            totnum[response] += 1
        else:
            totnum[response] = 1
    sortnum = sorted(totnum.items(), key=operator.itemgetter(1), reverse=True)
    return sortnum[0][0]
'''
if (filename == 'twitter.txt'):
    flabel = open(labelfilename)
    labellines = flabel.readlines()
    # print(labellines)
    list1 = []
    for lines in labellines:
        words = lines.split()
        list1.append(words)
    # print(list1)

    f = open(filename)
    dlines = f.readlines()
    # print(dlines)
    dataset = []
    for lines1 in dlines:
        words1 = lines1.split()
        dataset.append(words1)
    # print(dataset)
    for i in range(10):
        accuracycount = 0
        totalcount = 0
        testSet = []
        trainingSet = []
        labeltest = []
        labeltrain = []
        print(i)
        i1 = (i * int(len(dlines) / 10))
        i2 = ((i + 1) * int(len(dlines) / 10))
        # print(i1, i2)
        for j in range(len(dlines)):
            # if (j < int(len(dataset) / 10)):
            if (i1 <= j <= i2):
                testSet.append(dlines[j])
                labeltest.append(list1[j])
            else:
                trainingSet.append(dlines[j])
                labeltrain.append(list1[j])
        # print('test', testSet)
        # print('train', trainingSet)
        # trainingSet,testSet = gettraintest(trainingSet, testSet)
        vector = CountVectorizer()
        vector.fit(trainingSet)
        # print(vector.vocabulary_)
        features = vector.get_feature_names()
        # print(vector.get_feature_names())
        counts = vector.transform(trainingSet)
        # print(counts.shape)
        # print(counts.toarray())
        vectorizer = TfidfTransformer()
        vectorizer.fit(counts)
        # print(vectorizer.idf_)
        freq = vectorizer.transform(counts)
        # print(freq.toarray())
        testvect = vector.transform(testSet)
        testvecttfid = vectorizer.transform(testvect)
        trainingSet = counts.toarray()
        testSet = testvecttfid.toarray()
        # print('new train:', trainingSet)
        # print('new test:', testSet)
        neighbors, accuracycount, totalcount, tocalf1 = getneighbor(trainingSet, testSet, labeltest, list1, dataset,
                                                                    accuracycount, totalcount)
        print('Classification Report: ', classification_report(labeltest, tocalf1))
        accuracy = accuracycount / totalcount
        print(accuracy)
    # getneighbor(trainingSet,testSet)

else:
'''
def knnmain(filename, labelfilename):
    accuracycount = 0
    totalcount = 0
    lines = csv.reader(open(filename))
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
        neighbors, accuracycount, totalcount, tocalf1 = getneighbor(trainingSet,testSet, labeltest, list1, dataset, accuracycount, totalcount)
        print('Classification Report: ', classification_report(labeltest, tocalf1))
        accuracy = accuracycount / totalcount
        print(accuracy)

'''
filename = 'twitter._128.csv'
labelfilename = 'twitter_label.csv'
knnmain(filename, labelfilename)
'''