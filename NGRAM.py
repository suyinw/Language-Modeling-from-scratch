import string
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import time
import scipy

train = 'C:/Users/suyinw/Desktop/10707/Project 3/hw3/train.txt'
vali = 'C:/Users/suyinw/Desktop/10707/Project 3/hw3/val.txt'

HIDDEN_LAYER = 128
# OUTPUT_LAYER = 3500
LEARNING_RATE = 0.01
EPOCH_NUM = 100
VOC_SIZE = 8000
EMB_LAYER = 16
NGRAM = 4
BATCH_SIZE = 64
INPUT_LAYER = EMB_LAYER*(NGRAM-1)
randomSeed = np.random.RandomState(42)
mu, sigma = 0, 0.1

def preprocess(inputPath):
    files = open(inputPath, 'r').readlines()
    token = []
    docs = []
    for rows in files:
        rows = rows.lower()
        rows = 'START'+' '+rows+' '+'END'
        rows = rows.split()
        docs.append(rows)
        for val in rows:
            token.append(val)
    countToken = Counter(token)
    topToken = countToken.most_common(7999)
    diction = {}
    i = 0
    for tuples in topToken:
        key = tuples[0]
        diction[key] = i
        i +=1
    # diction['UNK'] = 7999
    return diction,docs

def ngram(diction,docs):
    # replace the unknown words to the original documents after truncation
    i = 0
    for lines in docs:
        j = 0
        for vals in lines:
            if vals not in diction:
                docs[i][j] = 'UNK'
            j+=1
        i +=1
    # construct 4 grams of the whole corpus and each document
    diction['UNK'] = 7999
    ngrams = []
    ngramRep = []
    # ngramIndMatrix = []
    j = 0
    for lines in docs:
        ngramRep.append([])
        for i in range(len(lines)-3):
            ngramPh = lines[i]+ ' '+ lines[i+1]+' '+ lines[i+2]+' '+lines[i+3]
            ngramRep[j].append(ngramPh)
            ngrams.append(ngramPh)
            # ngramIndMatrix.append(ngramIndex)
        j+=1
    ngramIndMatrix = np.zeros((len(ngrams),NGRAM),dtype=np.int64)
    for i in range(len(ngrams)):
        for j in range(NGRAM):
            phrases = ngrams[i].split()
            ngramIndMatrix[i,j] = diction[phrases[j]]

    # get the top 50 ngrams
    topNgram = Counter(ngrams)
    topfif = topNgram.most_common(50)
    topNgram = topNgram.most_common(len(ngrams))
    print topfif

    # prepare for drawing the distribution of 4 grams
    xphrase = []
    yvalues = []
    for items in topNgram:
        xphrase.append(items[0])
        yvalues.append(items[1])

    return topfif, xphrase,yvalues,ngrams,ngramRep,ngramIndMatrix


def making_distribution_graph(xphrase,yvalues):
    x = [i for i in range(len(xphrase))]
    x = np.array(x)
    yvalues = np.array(yvalues)
    # plt.xticks(x, xphrase)
    plt.plot(x, yvalues)
    plt.xlim([-20, max(x)+1])
    plt.ylim([-0.05, max(yvalues)+0.5])
    plt.show()


def initialization(INPUT_LAYER,HIDDEN_UNIT,VOC_SIZE,EMB_LAYER):
    w1 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT,INPUT_LAYER])
    w2 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT,VOC_SIZE])
    b1 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT,1])
    b2 = randomSeed.normal(mu, sigma,[VOC_SIZE,1])
    embed = randomSeed.normal(mu, sigma,[VOC_SIZE,EMB_LAYER])
    return w1,w2,b1,b2,embed


def preActivation(W,x,bias):
    a = W.dot(x.T)+ bias
    return a

def tanh_forward(z):
    return np.tanh(z)

def tanh_backward(z):
    return 1-np.square(tanh_forward(z))

def softmax(z):
    p = np.exp(z)
    prob = p/p.sum(axis=0)
    return prob


def getLoss(w1, w2, b1, b2, ngrams, diction, embed, ngramIndMatrix,linear=True):
    ## forward Propogation
    diction['UNK'] = 7999
    np.random.shuffle(ngramIndMatrix)
    training_data = ngramIndMatrix
    mini_batches = [
        training_data[k:k + BATCH_SIZE]
        for k in xrange(0, len(ngrams), BATCH_SIZE)]
    loss = 0
    perp = 0
    for mini_batch in mini_batches:
        index1 = mini_batch[:, 0]
        index2 = mini_batch[:, 1]
        index3 = mini_batch[:, 2]

        x1 = embed[index1, :]  # 64*16
        x2 = embed[index2, :]
        x3 = embed[index3, :]
        indexarray = np.concatenate((index1, index2, index3))
        unique = Counter(indexarray)
        uniqueDict = dict(unique)
        x = np.concatenate((x1, x2, x3), axis=1)
        # print x

        y = mini_batch[:, 3]
        # print y
        a1 = preActivation(w1, x, b1)
        if linear == True:
            hidden = a1
        else:
            hidden = tanh_forward(a1)
        a2 = preActivation(w2.T, hidden.T, b2)
        soft = softmax(a2)
        values = []
        for i in range(len(y)):
            values.append(soft[y[i], i])

        loss -= np.sum(np.log(values))
        # perp -= np.sum(np.log2(values))

    meanLoss = loss / len(training_data)
    # perpl = perp / len(training_data)
    meanPerplex = np.exp(meanLoss)

    return meanLoss,meanPerplex


def LanguageModel(w1,w2,b1,b2,ngrams,ngramsVali,diction,embed,ngramIndMatrix,ngramIndMatrixVali,linear = False):
    ## forward Propogation
    diction['UNK'] = 7999

    trainLoss = []
    valiLoss = []
    trainPerp = []
    valiPerp = []
    for i in range(EPOCH_NUM):
        print i
        np.random.shuffle(ngramIndMatrix)
        training_data = ngramIndMatrix
        mini_batches = [
            training_data[k:k + BATCH_SIZE]
            for k in xrange(0,len(ngrams),BATCH_SIZE)]
        loss = 0
        perp = 0
        start_time = time.time()
        for mini_batch in mini_batches:

            index1 = mini_batch[:, 0]
            index2 = mini_batch[:, 1]
            index3 = mini_batch[:, 2]

            x1 = embed[index1,:]  #64*16
            x2 = embed[index2,:]
            x3 = embed[index3,:]
            indexarray = np.concatenate((index1,index2,index3))
            unique = Counter(indexarray)
            uniqueDict = dict(unique)
            x = np.concatenate((x1,x2,x3),axis = 1)
            # print x

            y = mini_batch[:,3]
            # print y
            a1 = preActivation(w1, x, b1)
            if linear == True:
                hidden = a1
            else:
                hidden = tanh_forward(a1)
            a2 = preActivation(w2.T, hidden.T, b2)
            soft = softmax(a2)
            values =[]
            onehot = np.zeros((soft.shape))
            for i in range(len(y)):
                values.append(soft[y[i],i])
                onehot[y[i],i] = 1
            # ##calculate loss
            loss -= np.sum(np.log(values))
            # perp -= np.sum(np.log2(values))

            ## back propograte
            dLdAhat = soft-onehot  # 8000*64
            dLdW2 = np.dot(dLdAhat,hidden.T)  # 8000*128
            dLdb2 = dLdAhat #c = hid_bias  8000*64
            dLdhx = np.dot(w2,dLdAhat)  # 128*64
            # print 'at gram',t,'hidden layer dLdhx is',dLdhx
            if linear == True:
                dLdA = dLdhx  # 128*64
            else:
                dLdA = np.multiply(dLdhx,tanh_backward(a1)) #128*64
            dLdw1 = np.dot(dLdA,x) #128*48
            dLdb1 = dLdA  #128*64
            dLdx = np.dot(w1.T,dLdA).T# 64*48
            dLdw01 = dLdx[:,0:EMB_LAYER]
            dLdw02 = dLdx[:,EMB_LAYER:2*EMB_LAYER]
            dLdw03 = dLdx[:,2*EMB_LAYER:3*EMB_LAYER]
            ## update parameters
            w1 += -LEARNING_RATE * dLdw1/BATCH_SIZE  #128*48
            w2 += -LEARNING_RATE * dLdW2.T/BATCH_SIZE #128*8000
            # print 'at gram', t,'weight update is',w1,w2
            mdLdb1 = np.mean(dLdb1,axis = 1).reshape((b1.shape))
            mdLdb2 = np.mean(dLdb2, axis=1).reshape((b2.shape))
            b1 += -LEARNING_RATE * mdLdb1  #128*1
            b2 += -LEARNING_RATE * mdLdb2  #8000*1
            # print 'at gram', t, 'bias update is',b1,b2
            embed[index1, :] += -LEARNING_RATE * dLdw01
            embed[index2, :] += -LEARNING_RATE * dLdw02
            embed[index3, :] += -LEARNING_RATE * dLdw03
            meanIndex = uniqueDict.keys()
            wordCount = uniqueDict.values()
            for i in range(len(meanIndex)):
                embed[meanIndex[i]] = embed[meanIndex[i]]/wordCount[i]

        meanLoss = loss/ len(training_data)
        # perpl = perp /len(training_data)
        meanPerplex = np.exp(meanLoss)
        # print meanLoss,meanPerplex
        trainLoss.append(meanLoss)
        trainPerp.append(meanPerplex)
        meanvaliloss,valiper = getLoss(w1, w2, b1, b2, ngramsVali, diction, embed, ngramIndMatrixVali, linear=False)
        valiLoss.append(meanvaliloss)
        valiPerp.append(valiper)
        print 'train',meanLoss,meanPerplex,'validation',meanvaliloss,valiper
        print("--- %s seconds ---" % (time.time() - start_time))

    return trainLoss,trainPerp,valiLoss,valiPerp,embed,w1, w2, b1, b2

# def predictWords(WordNum,):
#


def predict(w1,w2,b1,b2,diction,ngrams,nword,embed,ngramIndMatrix):
    ngramDict = {grams: i for i, grams in enumerate(ngrams)}
    ngramDictRe = {value: key for key, value in ngramDict.iteritems()}
    wordDict = {value: key for key, value in diction.iteritems()}
    index = [1,81835,4,82454,83427]
    print [ngramDictRe[i] for i in index]
    mini_batch = ngramIndMatrix[index]
    genText =[]
    for j,word in enumerate(mini_batch):
        sentence = []
        newWord = word
        for i in range(nword):
            # newWord = newWord
            index1 = newWord[0]
            index2 = newWord[1]
            index3 = newWord[2]
            x1 = embed[index1, :]  # 64*16
            x2 = embed[index2, :]
            x3 = embed[index3, :]
            x = np.concatenate((x1, x2, x3))
            x = x.reshape((len(x),1)).T

            a1 = preActivation(w1, x, b1)
            hidden = tanh_forward(a1)
            a2 = preActivation(w2.T, hidden.T, b2)
            soft = softmax(a2)
            predictWID = np.argmax(soft)
            word = wordDict[predictWID]
            # print j,i,word
            newWord = [index2,index3,predictWID]
            sentence.append(word)
        # print sentence
        genText.append(sentence)
    print genText
    return genText


def making_loss_graph(lossTrain,lossVal):
    lossTrain = np.array(lossTrain)
    lossVal = np.array(lossVal)

    maxval = max(np.max(lossTrain),np.max(lossVal))
    epoch = np.array(range(EPOCH_NUM))
    trainline, = plt.plot(epoch,lossTrain,'r',label='training')
    valiLine, = plt.plot(epoch,lossVal, 'g',label='validation')

    plt.legend(handles=[trainline, valiLine])

    plt.title('Cross Entropy Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim([-0.05, EPOCH_NUM])
    plt.ylim([-0.05, maxval+2])

def making_perp_graph(trainAcc,valiAcc):
    trainAcc = np.array(trainAcc)
    valiAcc = np.array(valiAcc)
    maxval = max(np.max(trainAcc), np.max(valiAcc))
    epoch = np.array(range(EPOCH_NUM))
    trainline, = plt.plot(epoch,trainAcc,'r',label='training')
    valiLine, = plt.plot(epoch,valiAcc, 'g',label='validation')

    plt.legend(handles=[trainline, valiLine])

    plt.title('Perplexity')
    plt.xlabel('epochs')
    plt.ylabel('Perplexity')
    plt.xlim([-0.05, EPOCH_NUM])
    plt.ylim([-0.05, maxval+2])


def visualizeEmb(embed,diction):
    index = np.array(randomSeed.randint(low = 0,high=len(embed),size = 500))
    sample = embed[index,:]  #500*2
    wordDict = {value:key for key,value in diction.iteritems()}
    words = [wordDict[i] for i in index]

    x = sample[:,0]   #500*1
    y = sample[:,1]
    plt.scatter(x, y)

    for i, word in enumerate(words):
        plt.annotate(word, (x[i]+0.1,y[i]+0.1))

    plt.title("Word Embeddings Visualization")
    plt.xlabel("Embedding 1")
    plt.ylabel("Embedding 2")
    plt.show()




def main():
    diction, docs = preprocess('C:/Users/suyinw/Desktop/10707/Project 3/hw3/train.txt')
    dictionvali, docsvali = preprocess('C:/Users/suyinw/Desktop/10707/Project 3/hw3/val.txt')
    topfif, xphrase, yvalues, ngrams, ngramRep,ngramIndMatrix = ngram(diction, docs)
    topfifV, xphraseV, yvaluesV, ngramsVali, ngramRepVali,ngramIndMatrixVali = ngram(diction, docsvali)
    # making_distribution_graph(xphrase, yvalues)
    w1, w2, b1, b2, embed = initialization(INPUT_LAYER, HIDDEN_LAYER, VOC_SIZE, EMB_LAYER)
    trainLoss, trainPerp, valiLoss, valiPerp, embed,w1, w2, b1, b2 = LanguageModel(w1,w2,b1,b2,ngrams,ngramsVali,diction,embed,ngramIndMatrix,ngramIndMatrixVali,linear = False)
    # visualizeEmb(embed, diction)
    genText = predict(w1, w2, b1, b2, diction, ngrams, 10, embed, ngramIndMatrix)
    making_loss_graph(trainLoss,valiLoss)
    making_perp_graph(trainPerp, valiPerp)
main()