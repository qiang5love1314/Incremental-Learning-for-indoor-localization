import os
import numpy as np
import pandas as pd
import time
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from scipy import signal
from numpy import random
from sklearn.preprocessing import LabelBinarizer, Normalizer
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from BroadLearningSystem import *

from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping
from keras import regularizers, optimizers
import cv2

'CNN processing'
def Zscorenormalization(x):         #归一化
    x = ( x - np.mean(x)) / np.std(x)
    return x

def getXlabel():
    xLabel = []
    for i in range(21):     #横坐标
        str = '%d' % (i + 1)
        xLabel.append(str)
    return xLabel

def getYlabel():
    yLabel = []
    for j in range(23):     #纵坐标
        if(j<9):
            num=0
            str= '%d%d' % (num, j+1)
            yLabel.append(str)
        else:
            yLabel.append('%d'% (j+1) )
    return yLabel

def rawCSI():
    xLabel = getXlabel()
    yLabel = getYlabel()
    count = 0
    originalCSI=np.zeros((317, 135000), dtype=np.float)
    newName = []
    label = np.empty((0, 2), dtype=np.int)

    for i in range(21):
        for j in range(23):
            filePath = r"C:\Users\dell\Desktop\47SwapData\coordinate" + xLabel[i] + yLabel[j] + ".mat"
            name = xLabel[i] + yLabel[j]
            if (os.path.isfile(filePath)):
                c = loadmat(filePath)
                CSI = np.reshape(c['myData'], (1, 3 * 30 * 1500))
                originalCSI[count, :] = CSI
                newName.append(name)
                label = np.append(label, [[int(xLabel[i]), int(yLabel[j])]], axis=0)
                count += 1
    return originalCSI, label

def generateImage():
    midString = []
    xLabel = getXlabel()
    yLabel = getYlabel()
    datalist = []

    count = 0
    for i in range(21):
        for j in range(23):
            filePath = r'C:\Users\dell\Desktop\47SwapData\coordinate' + xLabel[i] + yLabel[j] + '.mat'
            if (os.path.isfile(filePath)):
                count +=1
                data = loadmat(filePath)
                test = data['myData']  # 3*30*1500
                for k in range(30):
                    swap = test[:, :, 30 * (k):30 * (k + 1)]
                    newd = np.rollaxis(swap, 0, 3)  # change channel, 30*30*3
                    amplitude = abs(np.array(newd))  #
                    datalist.append(amplitude)

    Label = []
    numOfImage = 30
    count = 0
    for i in range(21):
        for j in range(23):
            filePath = r"C:\Users\dell\Desktop\47SwapData\coordinate" + xLabel[i] + yLabel[j] + ".mat"
            if (os.path.isfile(filePath)):
                count += 1
                Label.append(numOfImage*[count])
    Label = np.reshape(Label, (317*numOfImage, 1)).flatten()
    return datalist, Label

def phaseMatrix():
    xLabel = getXlabel()
    yLabel = getYlabel()
    data = []
    phaseLabel = []
    midString = [] #str(list(range(1,50)))
    numOfImage = 30
    for i in range(numOfImage):
        str = '%d' % (i + 1)
        midString.append(str)

    for i in range(21):
        for j in range(23):
            for k in range(numOfImage):
                filePath = r"C:\Users\dell\Desktop\47imaginary_part\image-phase-3D\Phase" + xLabel[i] + yLabel[
                           j] + "-" + midString[k] + ".jpg"
                if (os.path.isfile(filePath)):
                    image = cv2.imread(filePath)
                    image = cv2.resize(image,(30,30))
                    image = img_to_array(image)
                    data.append(image)

    count = 0
    for i in range(21):
        for j in range(23):
            filePath = r"C:\Users\dell\Desktop\47SwapData\coordinate" + xLabel[i] + yLabel[j] + ".mat"
            if (os.path.isfile(filePath)):
                count += 1
                phaseLabel.append(numOfImage*[count])
    phaseLabel = np.reshape(phaseLabel, (317*numOfImage, 1)).flatten()
    return data, phaseLabel

def constructCNN(traindata, testdata, trainlabel, testlabel):
    inputShape = (30, 30, 3)
    chanDim = -1

    'conv > relu > pool'
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    #'(conv > relu) * 2 > pool'    , kernel_regularizer=regularizers.l2(0.01)
    model.add(Conv2D(64, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    '(conv > relu) * 2 > pool'
    model.add(Conv2D(128, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    'FC > rely'
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.25))

    'classier > softmax'
    model.add(Dense(317, activation='softmax'))

    'sgd > compile'
    print("compiling model...")
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    history = model.fit(traindata, trainlabel, epochs=100, batch_size=50, validation_data=(testdata, testlabel))    # , callbacks = [EarlyStopping(monitor='val_acc', patience=2)] - loss: 0.9583 - acc: 0.7375  - val_loss: 1.4746 - val_acc: 0.6685
    model.save(filepath=r'D:\pythonWork\indoor Location\third-code\model-lab-Amplitude.h5')
    # model.save_weights('model_weights-v15.h5')
    plotModelHistory(history)

def Zscorenormalization(x):         #归一化
    x = ( x - np.mean(x)) / np.std(x)
    return x

def main():
    print('generate dataset')
    originalCSI, labelplus = generateImage()   # 原始振幅数据集与标签(317*30, 30, 30, 3)，(317*30, 1)
    other, label = rawCSI()
    data = Zscorenormalization(np.array(originalCSI, dtype='float') )

    labels = np.array(labelplus)
    lb = LabelBinarizer()
    phaseLabel = lb.fit_transform(labels)
    traindata, testdata, trainlabel, testlabel = train_test_split(data, phaseLabel, test_size=0.2, random_state=10)


    lbtestlabel = lb.inverse_transform(testlabel)
    print(len(traindata),len(testdata))

    'train and test CNN model'
    constructCNN(traindata, testdata, trainlabel, testlabel)

    'cluster the predicts of phase images for every position'
    from sklearn.cluster import KMeans
    Mymodel = load_model(r'D:\pythonWork\indoor Location\third-code\model-lab-Amplitude.h5')
    predict = Mymodel.predict_classes(testdata)
    cluster = []
    for i in range(317):
        list1 = np.where(lbtestlabel == (i+1))
        index = predict[list1]
        position = label[index]
        swap = []
        for i in range(len(position)):
            first = [position[i][0],position[i][1]]
            swap.append(first)
        if swap != []:
            kmeans = KMeans(n_clusters=1, random_state=0).fit(swap)
            cluster_center = kmeans.cluster_centers_
            cluster.append(cluster_center)
        else:
            cluster.append(np.reshape(label[i],(1,2)))
    predictOfPhaseImage= np.reshape(cluster,(317,2))

    'BLS classification regression'
    N1 = 30  # # of nodes belong to each window
    N2 = 5  # # of windows -------Feature mapping layer
    N3 = 100  # # of enhancement nodes -----Enhance layer
    L = 15  # # of incremental steps
    M1 = 100  # # of adding enhance nodes
    M2 = 100  # # of adding feature mapping nodes
    M3 = 100  # # of adding enhance nodes
    s = 0.8  # shrink coefficient
    C = 2 ** -15  # Regularization coefficient
    print('-------------------BLS_BASE---------------------------')
    traindata = np.reshape(traindata, (len(traindata), 30 * 30 * 3))
    testdata = np.reshape(testdata, (len(testdata), 30 * 30 * 3))
    test_acc,test_time,train_acc_all,train_time,OutputOfTest = BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1,M2,M3)

    '找到每个类别的5个最大分类概率，进行初步期望回归，迭代次数为测试集长度'
    OutputOfTest = np.array(OutputOfTest)
    max_position = np.zeros((len(testdata), 5))
    max_position_weight = np.zeros((len(testdata), 5))
    list2 = []
    import heapq as hq
    for i in range(len(OutputOfTest)):
        maxpro = hq.nlargest(5, OutputOfTest[i])
        max_position_weight[i] = maxpro
        maxpsi = hq.nlargest(5, range(len(OutputOfTest[i])), OutputOfTest[i].__getitem__)
        max_position[i] = maxpsi
        result = 0
        for j in range(5):
            a = (np.multiply(max_position_weight[i][j], label[int(max_position[i][j])]))
            result += a
        list2.append(result)

    '找到测试集对应的坐标，每个采样点有[1, n]个初步期望回归，求平均，精度略有提升'
    list3 = []
    lbtestlabel = np.array(lbtestlabel)
    for i in range(317):
        index1 = np.where(lbtestlabel == (i+1))
        list4 = []
        for i in range(len(index1[0])):
            positionll = list2[index1[0][i]]
            list4.append(positionll)
            predictPositon = np.mean(list4, axis=0)
        list3.append(predictPositon)

    '将BLS计算的每个采样点最大分类概率作为权重w，与CNN聚类结果进行联合定位(1-w)，w的值很小'
    listPro = []
    for i in range(len(OutputOfTest)):
        probability = np.max(OutputOfTest[i])
        listPro.append(probability)
    list6 = []
    for i in range(317):
        index1 = np.where(lbtestlabel == (i + 1))
        list5 = []
        for i in range(len(index1[0])):
            position = listPro[index1[0][i]]
            list5.append(position)
            newPro = np.max(list5)
        list6.append(newPro)

    '--------estimation location----------'
    list7 = []
    for i in range(317):
        error = np.multiply(np.array(list3[i]), np.array(list6[i])) + np.multiply((1-np.array(list6[i])), predictOfPhaseImage[i])
        list7.append(error)
    accuracyLab(list7, label)   #   2.3800846844582346 m
    # saveTestErrorMat(list7, label, 'Lab-Amplitude-Error')


def accuracyLab(label1, label2):
    error = np.asarray(label1 - label2)
    print(np.mean(np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)) * 50 / 100 , 'm')

def saveTestErrorMat(predictions , testlabel, fileName):
    error = np.asarray(predictions - testlabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 50 / 100
    savemat(fileName+'.mat', {'array': sample})

def reloadModel(filePath, testdata, testlabel):
    model = load_model(filePath)
    acc = model.evaluate(testdata, testlabel, batch_size=50)
    print('loss = '+ str(acc[0]))
    print('accuracy =' + str(acc[1]))

def plotModelHistory(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()

if __name__ == '__main__':
    main()
    pass