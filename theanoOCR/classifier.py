# import the type of model
from keras.models import Sequential
# import layers
from keras.layers.core import Dense, Dropout, Activation, Flatten
# import convolution layers
from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras.models import load_model
# import matplotlib
import numpy as np
# import theano
from numpy import *
from PIL import Image
import os
from itertools import cycle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc
# import pandas as pd
import matplotlib.pyplot as plt

from theanoOCR import preprocessor

import time


class Classifier:
    def __init__(self, rows, cols):
        self.path = os.path.join(os.path.dirname(__file__), 'Original_letters')
        self.path3 = os.path.join(os.path.dirname(__file__), 'Intermidiate_Step')
        self.path2 = os.path.join(os.path.dirname(__file__), 'Original_input_letters')
        self.listing = None
        self.num_samples = None
        self.inlist = None
        self.label = None
        self.img_rows, self.img_cols = rows, cols
        self.batch_size = 32
        self.nb_classes = 9
        # no of rotations
        self.nb_epoch = 1
        self.img_channels = 1

        self.no_filters = 32
        self.no_pool = 2
        self.no_conv = 3

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def read_original_data(self):
        self.listing = os.listdir(self.path)
        self.num_samples = size(self.listing)
        self.listing.sort()

    def read_input_data(self):
        self.inlist = os.listdir(self.path2)
        # self.num_samples = size(self.listing)
        self.inlist.sort()
        # print(len(self.inlist))

    def preprocess(self):

        for file in self.listing:
            #     # print(file)
            im = Image.open(self.path + '/' + file)
            img = im.resize((self.img_rows, self.img_cols))
            #     gray = img.convert('L')
            img.save(self.path3 + "/" + file, 'JPEG')

        preprocessor.pre_processing()
        time.sleep(5)

    def create_matrix(self):
        imgmatrix = array([array(Image.open(self.path2 + "/" + im2)).flatten() for im2 in self.inlist], 'f')
        return imgmatrix

    def initialize_data(self):
        self.label = np.ones((self.num_samples,), dtype=int)
        self.label[0:103] = 0
        self.label[103:202] = 1
        self.label[202:291] = 2
        self.label[291:340] = 3
        self.label[340:396] = 4
        self.label[396:515] = 5
        self.label[515:616] = 6
        self.label[616:658] = 7
        self.label[658:] = 8

        imgmatrix = self.create_matrix()
        data, Label = shuffle(imgmatrix, self.label, random_state=4)
        train_data = [data, Label]

        (x, y) = (train_data[0], train_data[1])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=4)

        self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        self.x_train /= 255
        self.x_test /= 255

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.nb_classes)

    def create_model(self):
        model = Sequential()

        model.add(Convolution2D(self.no_filters, self.no_conv, self.no_conv, border_mode='valid',
                                input_shape=(1, self.img_rows, self.img_rows)))

        convout1 = Activation('relu')
        model.add(convout1)
        model.add(Convolution2D(self.no_filters, self.no_conv, self.no_conv))

        convout2 = Activation('relu')
        model.add(convout2)
        model.add(MaxPooling2D(pool_size=(self.no_pool, self.no_pool)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128))

        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))

        model.add(Activation('softmax'))

        return model

    def model_compile(self, model):

        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        return model
        # model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1,
        # validation_split=0.2)

    def model_train(self, model):
        model.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=1,
                  validation_data=(self.x_test, self.y_test))
        return model

    def get_prediction(self, model):
        y_score = model.predict(self.x_test)
        return y_score

    def get_evaluation(self, model):
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        return score

    def test_model(self, model):
        character = ["ග", "හ", "ක", "ල", "ම", "ප", "ස", "ත", "ය"]
        evaluation = dict()
        prediction = model.predict_classes(self.x_test[0:])
        real = self.y_test[0:]
        real = real.tolist()
        print(type(real[0][0]))
        # print(real)

        for x in range(0, len(prediction)):

            if prediction[x] not in evaluation:
                evaluation[prediction[x]] = [0, 0]
                predict_class = real[x].index(1.0)

                if predict_class not in evaluation:
                    evaluation[predict_class] = [0, 0]

                evaluation[predict_class][0] += 1
                if prediction[x] == predict_class:
                    evaluation[prediction[x]][1] += 1

            elif prediction[x] in evaluation:
                predict_class = real[x].index(1.0)

                if predict_class not in evaluation:
                    evaluation[predict_class] = [0, 0]

                evaluation[predict_class][0] += 1
                if prediction[x] == predict_class:
                    evaluation[prediction[x]][1] += 1

        for key, value in evaluation.items():
            print(character[key], "  ---> class ", key, "accuracy rate ", (value[1] / value[0]) * 100, "%")

        print(evaluation)

    def save_model(self, model):
        fname = os.path.join(os.path.dirname(__file__), 'Ancient_Classifier.h5')
        model.save(fname)

    def loadModel(self):
        fname = os.path.join(os.path.dirname(__file__), 'Ancient_Classifier.h5')
        return load_model(fname)

    def draw_roc_curve(self, y_score):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(0, self.nb_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.nb_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.nb_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.nb_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'red'])
        for i, color in zip(range(self.nb_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()


    def getNewModel(self):
        myModel = self.create_model()
        myModel = self.model_compile(myModel)
        return self.model_train(myModel)


myClassifier = Classifier(200, 200)
myClassifier.read_original_data()
myClassifier.read_input_data()
#     # myClassifier.preprocess()
myClassifier.initialize_data()

#     # test_score = myClassifier.get_prediction(myModel)
#     # myClassifier.draw_roc_curve(test_score)
"""Comment one of following lines interchangeably to train model or to load saved one """
# myModel = myClassifier.getNewModel()
myModel=myClassifier.loadModel()
myClassifier.test_model(myModel)
myClassifier.get_evaluation(myModel)
# myClassifier.save_model(myModel)

# fname = "/media/nishan/Entertainment/CNN/weights-Test-CNN.hdf5"
# model.save_weights(fname)
