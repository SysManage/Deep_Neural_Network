# import the type of model
# import layers
# import convolution layers

import os

from PIL import Image
from keras.models import load_model
from numpy import *


class Predictor:
    def __init__(self, rows, cols):

        self.test_path = os.path.join(os.path.dirname(__file__), 'seg_images')
        self.raw_listing = None
        self.num_samples = None
        self.traininglist = None
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



    def init_test_data(self):
        names=os.listdir(self.test_path)
        print(names)
        self.x_test= array([array(Image.open(self.test_path + "/" + im2)).flatten() for im2 in names ], 'f')
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
        self.x_test = self.x_test.astype('float32')
        self.x_test /= 255


    def get_prediction(self, model):
        y_score = model.predict(self.x_test)
        return y_score


    def test_model(self, model):
        character = ["ග", "හ", "ක", "ල", "ම", "ප", "ස", "ත", "ය"]
        evaluation = dict()
        prediction = model.predict_classes(self.x_test[0:])
        print(prediction)
        for predic in prediction:
            print(character[predic], end="")

        # real = self.y_test[0:]
        # real = real.tolist()
        # print(type(real[0][0]))
        # # print(real)
        #
        # for x in range(0, len(prediction)):
        #
        #     if prediction[x] not in evaluation:
        #         evaluation[prediction[x]] = [0, 0]
        #         predict_class = real[x].index(1.0)
        #
        #         if predict_class not in evaluation:
        #             evaluation[predict_class] = [0, 0]
        #
        #         evaluation[predict_class][0] += 1
        #         if prediction[x] == predict_class:
        #             evaluation[prediction[x]][1] += 1
        #
        #     elif prediction[x] in evaluation:
        #         predict_class = real[x].index(1.0)
        #
        #         if predict_class not in evaluation:
        #             evaluation[predict_class] = [0, 0]
        #
        #         evaluation[predict_class][0] += 1
        #         if prediction[x] == predict_class:
        #             evaluation[prediction[x]][1] += 1
        #
        # for key, value in evaluation.items():
        #     print(character[key], "  ---> class ", key, "accuracy rate ", (value[1] / value[0]) * 100, "%")
        #
        # print(evaluation)

    def loadModel(self):
        fname = os.path.join(os.path.dirname(__file__), 'Ancient_Classifier.h5')
        return load_model(fname)



predictor = Predictor(200, 200)
predictor.init_test_data()
myModel = predictor.loadModel()
predictor.test_model(myModel)