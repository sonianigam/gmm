#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from gmm_est import gmm_est
import scipy.stats
import random
import matplotlib.patches as mpatches


def main():
    """
    This function runs your code for problem 3.

    You can use this code for problem 4, but make sure you do not
    interfere with what you need to do for problem 3.
    """
    file_path = sys.argv[1]
    file = open(file_path)

    #ignore title
    first_line = file.readline()

    #read the remaining data within the file
    body = file.readlines()

    total = 0.0
    X = []
    Y = []

    for line in body:
        values = line.strip().split(',')
        if values[1] == '1':
            total += 1
        X.append(float(values[0]))
        Y.append(float(values[1]))

    p1 = total/len(X)

    classified_data = gmm_classify(X, [9.7748859236586476, 29.582587182965749], [21.922804563645315, 9.7837696128028284], [0.59765463038822264, 0.40234536961263107], [-24.822751728709839, -5.0601582832398666, 49.62444471952756], [7.9473354076752969, 23.322661814417472, 100.02433750441168], [0.20364945852681876, 0.49884302379613926, 0.29750751767685851], p1)

    # class1_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 1.
    class1_data = []
    class2_data = []

    for i in xrange(len(classified_data)):
        if classified_data[i] == 1:
            class1_data.append(X[i])
        else:
            class2_data.append(X[i])

    print 'Class 1'
    print class1_data

    # class2_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 2.
    print '\nClass 2'
    print class2_data

    X = np.array(X)
    Y = np.array(Y)

    class1 = X[np.nonzero(Y==1)[0]]
    class2 = X[np.nonzero(Y==2)[0]]
    blue_patch = mpatches.Patch(color='blue', label='Class 1')
    red_patch = mpatches.Patch(color='red', label='Class 2')

    bins = 60 # the number 50 is just an example. 
    plt.title("Class One and Class Two: Actuals and Predicted")
    plt.hist(class1, bins, alpha =0.5, color = 'blue') 
    plt.hist(class2, bins, alpha =0.5, color = 'red') 
    plt.legend(handles=[blue_patch, red_patch])
    plt.scatter(X, [-5]*len(X),  c=classified_data, alpha=0.5)

    plt.savefig('Classify.png')



    #problem 4
    prior_classified_data = np.zeros(len(X))

    for i in xrange(len(X)):
        value = random.uniform(0.0, 1.0)
        if value < p1:
            prior_classified_data[i] = 1
        else:
            prior_classified_data[i] = 2

    error = 0
    prior_error = 0

    for i in xrange(len(X)):
        if classified_data[i] != Y[i]:
            error += 1
        if prior_classified_data[i] != Y[i]:
            prior_error += 1


    error = error/ len(X)
    prior_error = prior_error/ len(X)

    #THIS IS FOR PROBLEM 4'S OUTPUT!!
    # print "The GMM Model's Error: " + str(error)
    # print "The Prior Probability Model's Error: " + str(prior_error)




def gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1):
    """
    Input Parameters:
        - X           : N 1-dimensional data points (a 1-by-N numpy array)
        - mu1         : means of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - sigmasq1    : variances of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - wt1         : weights of Gaussian components of the 1st class (a 1-by-K1 numpy array, sums to 1)
        - mu2         : means of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - sigmasq2    : variances of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - wt2         : weights of Gaussian components of the 2nd class (a 1-by-K2 numpy array, sums to 1)
        - p1          : the prior probability of class 1.

    Returns:
        - class_pred  : a numpy array containing results from the gmm classifier
                        (the results array should be in the same order as the input data points)
    """
    N = len(X)
    class_pred = np.zeros(N)
    k1 = len(mu1)
    k2 = len(mu2)

    for i in xrange(N):
        prob_one = -10
        prob_two = -10

        for x_1 in range(k1):
            value = wt1[x_1] * scipy.stats.norm(mu1[x_1], np.sqrt(sigmasq1[x_1])).pdf(X[i])
            if value > prob_one:
                prob_one = value

        for x_2 in range(k2):
            value = wt2[x_2] * scipy.stats.norm(mu2[x_2], np.sqrt(sigmasq2[x_2])).pdf(X[i])
            if value > prob_two:
                prob_two = value        

        if prob_one >= prob_two:
            class_pred[i] = 1
        else:
            class_pred[i] = 2

    return class_pred


def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []

    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
        else:
            X2.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)

    return X1, X2

if __name__ == '__main__':
    main()
