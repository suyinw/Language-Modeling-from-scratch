# -*- coding: utf-8 -*-
import numpy as np
import math
import scipy.io
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

INPUT_LAYER = 784
HIDDEN_LAYER = 100
OUTPUT_LAYER = 10
LEARNING_RATE = 0.01
EPOCH_NUM = 300

def getData(filename):
    files = open(filename, 'r').readlines()
    xtrain = []
    ytrain = []
    for lines in files:
        lines = lines.rstrip()
        value = lines.split(',')
        value = [float(val) for val in value]
        xtrain.append(value[:-1])
        ytrain.append(value[-1])
    xtrain = np.matrix(xtrain)
    ytrain = np.array(ytrain)
    return xtrain,ytrain

# xtrain, ytrain = getData('digitstrain.txt')
def getOnehot(ytrain):
    onehotMatrix = np.zeros((ytrain.shape[0], OUTPUT_LAYER))
    for i in range(ytrain.shape[0]):
        for j in range(OUTPUT_LAYER):
            if ytrain[i] == j:
                onehotMatrix[i][j] = 1
    return onehotMatrix


def initialization(INPUT_LAYER,HIDDEN_LAYER):
    rangeVal = float(np.sqrt(6) / np.sqrt(INPUT_LAYER + HIDDEN_LAYER))
    w1 = np.random.uniform(-rangeVal, rangeVal, [HIDDEN_LAYER, INPUT_LAYER])
    b1 = np.zeros((HIDDEN_LAYER, 1))
    w2 = np.random.uniform(-rangeVal, rangeVal, [HIDDEN_LAYER, OUTPUT_LAYER])
    b2 = np.zeros((OUTPUT_LAYER, 1), dtype=float)
    return w1,w2,b1,b2

def preActivation(W,x,bias):
    a = W.dot(x.T)+ bias
    return a

def sigmoid_forward(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_backward(x):
    return np.multiply(sigmoid_forward(x),(1-sigmoid_forward(x)))

def softmax(z):
    p = np.exp(z)
    prob = p/p.sum(axis=0)
    return prob

def getLoss(w1,w2,b1,b2,xtrain,ytrain):
    loss = 0
    count = 0
    for t in range(xtrain.shape[0]):
        # for t in range(5):
        x = xtrain[t]
        y = ytrain[t]
        a = preActivation(w1, x, b1)
        hidden = sigmoid_forward(a)
        preact = preActivation(w2.T, hidden.T, b2)
        soft = softmax(preact)
        ##calculate loss
        loss += -np.log(soft[y])
        maxIndex = np.argmax(soft)

        if int(y) == int(maxIndex):
            count += 1
    meanLoss = loss.item()/xtrain.shape[0]
    accuracy = float(count) / int(xtrain.shape[0])

    return meanLoss,accuracy

def NeuralNet(w1,w2,b1,b2,xtrain,xval,xtest,ytrain,yval,ytest,onehotMatrix):
    ## forward Propogation
    trainLoss = []
    valiLoss = []
    testLoss = []
    trainAcc = []
    valiAcc = []
    testAcc = []
    for i in range(EPOCH_NUM):
        loss = 0
        count = 0
        for t in range(xtrain.shape[0]):
            # for t in range(5):
            x = xtrain[t] # 1*784
            y = ytrain[t]  #1*1
            onehot = onehotMatrix[t].reshape((1,10)) # 1*10
            a = preActivation(w1, x, b1)  #100*1
            hidden = sigmoid_forward(a) # 100*1
            preact = preActivation(w2.T, hidden.T, b2)  #10*1
            soft = softmax(preact) #10*1
            ##calculate loss
            loss += -np.log(soft[y])
            maxIndex = np.argmax(soft)

            ## back propograte
            dLdAhat = soft-onehot.T  # 10*1
            dLdW2 = np.dot(dLdAhat,hidden.T)  # 10*100
            dLdb2 = dLdAhat #c = hid_bias  784*1
            dLdhx = np.dot(w2,dLdAhat)  # 100*1

            dLdA = np.multiply(dLdhx,np.multiply(hidden,(1-hidden)))  #100*1
            dLdw1 = np.dot(dLdA,x)             #100*784
            dLdb1 = dLdA  #100*1
            ## update parameters
            w1 += -LEARNING_RATE * dLdw1
            w2 += -LEARNING_RATE * dLdW2.T

            b1 += -LEARNING_RATE * dLdb1
            b2 += -LEARNING_RATE * dLdb2
            if int(y) == int(maxIndex):
                count += 1
        meanloss = loss.item()/xtrain.shape[0]
        accuracy = float(count)/int(xtrain.shape[0])
        trainLoss.append(meanloss)
        trainAcc.append(accuracy)
        meanvaliloss,valiacu = getLoss(w1,w2,b1,b2,xval,yval)
        valiLoss.append(meanvaliloss)
        valiAcc.append(valiacu)
        meantestloss,testacu = getLoss(w1,w2,b1,b2,xtest,ytest)
        testLoss.append(meantestloss)
        testAcc.append(testacu)

        print i,'train',meanloss,accuracy,'validation',meanvaliloss,valiacu,'test',meantestloss,testacu

    return trainLoss,valiLoss,testLoss,trainAcc,valiAcc,testAcc

def making_loss_graph(lossTrain,lossVal,lossTest):
    lossTrain = np.array(lossTrain)
    lossVal = np.array(lossVal)
    lossTest = np.array(lossTest)

    maxval = max(np.max(lossTrain),np.max(lossVal),np.max(lossTest))
    epoch = np.array(range(EPOCH_NUM))
    trainline, = plt.plot(epoch,lossTrain,'r',label='training')
    testLine, = plt.plot(epoch,lossVal, 'g',label='test')
    valiLine, = plt.plot(epoch,lossTest, 'b',label='validation')

    plt.legend(handles=[trainline, valiLine, testLine])

    plt.title('Cross Entropy Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim([-0.05, EPOCH_NUM])
    plt.ylim([-0.05, maxval+2])
    return maxval

def making_accuracy_graph(trainAcc,valiAcc,testAcc,maxval):
    trainAcc = np.array(trainAcc)
    valiAcc = np.array(valiAcc)
    testAcc = np.array(testAcc)

    epoch = np.array(range(EPOCH_NUM))
    trainline, = plt.plot(epoch,trainAcc,'r',label='training')
    testLine, = plt.plot(epoch,valiAcc, 'g',label='test')
    valiLine, = plt.plot(epoch,testAcc, 'b',label='validation')

    plt.legend(handles=[trainline, valiLine, testLine])

    plt.title('Accuracy and Cross Entropy Loss')
    plt.xlabel('epochs')
    plt.ylabel('accuracy and cross entropy loss')
    plt.xlim([-0.05, EPOCH_NUM])
    plt.ylim([-0.05, maxval+0.5])


def mainNN(RBM = True,autoencoder = False,denoised = False):
    xtrain, ytrain = getData('digitstrain.txt')
    xval, yval = getData('digitsvalid.txt')
    xtest, ytest = getData('digitstest.txt')
    onehotMatrix = getOnehot(ytrain)
    w1, w2, b1, b2 = initialization(INPUT_LAYER, HIDDEN_LAYER)
    if RBM == True:
        Weight, hid_bias, obs_bias = RBMFinal.main()
        w1, b1 = Weight,hid_bias
    if autoencoder == True:
        w1, b1 = Autoencoder.main(denoise = False)
    if denoised == True:
        w1, b1 = Autoencoder.main(denoise = True)
    trainLoss, valiLoss, testLoss, trainAcc, valiAcc, testAcc = NeuralNet(w1,w2,b1,b2,xtrain,xval,xtest,ytrain,yval,ytest,onehotMatrix)
    maxval = making_loss_graph(trainLoss, valiLoss, testLoss)
    making_accuracy_graph(trainAcc, valiAcc, testAcc,maxval)

import time
start_time = time.time()
mainNN(RBM = False,autoencoder = True)
print("--- %s seconds ---" % (time.time() - start_time))
