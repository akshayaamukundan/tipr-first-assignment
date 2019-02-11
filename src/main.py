from sklearn.model_selection import cross_val_score
import csv
import random
import math
import operator
import nn as knn
import numpy as np
import projections as rp
import bayes as nb
import lsh as ls
import skbayes as sknb
import sknn as nnsk
import pca as pc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

print('Examples of the way filename has to be input for the 3 dataset:')
print('dolphins.csv')
print('pubmed.csv')
print('twitter.txt')
filename = input('input filename with extension without enclosing in single quotes')
print('Confirming filename: ',filename)
print('Example of the way labelfilename has to be input for the 3 dataset:')
print('dolphins_label.csv')
print('pubmed_label.csv')
print('twitter_label.txt')
labelfilename = input('input labelfilename with extension without enclosing in single quotes')
print('Confirming labelfilename: ',labelfilename)
print('type of data file is already considered in code')

if (filename == 'twitter.txt'):
    flabel = open('twitter_label.txt', 'rt')
    labellines = flabel.readlines()
    list1 = []
    for lines in labellines:
        words = lines.split()
        list1.append(words)
    newlabelfilename = 'twitter_label.csv'
    with open(newlabelfilename, 'w', newline='') as flabl:
        writer = csv.writer(flabl, delimiter=' ')
        writer.writerows(list1)
    flabl.close()

    f = open('twitter.txt', 'rt')
    dlines = f.readlines()
    dataset = []
    for lines1 in dlines:
        words1 = lines1.split()
        dataset.append(words1)

    vector = CountVectorizer()
    vector.fit(dlines)
    features = vector.get_feature_names()
    counts = vector.transform(dlines)
    countbow = counts.toarray()
    vectorizer = TfidfTransformer()
    vectorizer.fit(counts)
    freq = vectorizer.transform(counts)
    newfilename = 'twitter.csv'
    with open(newfilename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(freq.toarray())
    f.close()
    filename = 'twitter.csv'
    labelfilename = 'twitter_label.csv'

# Task 1
# Random Projections
print('Task1:')
rp.myprojmain(filename, labelfilename)
print('Reduced dimension of original data stored in csv format')

# Task 2 and Task 3
# k NN
print('Task2: and Task3:')
print('kNN')
knn.knnmain(filename, labelfilename)

# NB
print('NB')
nb.nbmain(filename,labelfilename)

#Task4
print('Task4:')
#GaussianNB
print('Results of GaussianNB classifier using sklearn')
sknb.sknbmain(filename, labelfilename)
#kNN
print('Results of kNN classifier using sklearn')
nnsk.sknnmain(filename, labelfilename)

#Task5
print('Task5:')
print('Comparison and respective plots are shown in report')

#Task6 and Task7
print('Task6 and Task7:')

# LSH
print('LSH')
ls.lshmain(filename, labelfilename)
#PCA
print('PCA and results:')
dataset = pc.pcamain(filename, labelfilename)