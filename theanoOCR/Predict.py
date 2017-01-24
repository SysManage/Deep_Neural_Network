# import the type of model
# import layers
# import convolution layers

from PIL import Image
from keras.models import load_model
from numpy import *


class Predictor:
    def __init__(self, rows, cols):

        self.test_path = os.path.join(os.path.dirname(__file__), 'seg_images')
        self.num_samples = None
        self.label = None
        self.img_rows, self.img_cols = rows, cols
        self.nb_classes = 9

        self.img_channels = 1

        self.x_test = None




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
        prediction = model.predict_classes(self.x_test[0:])
        print(prediction)
        for predic in prediction:
            print(character[predic], end="")


    def loadModel(self):
        fname = os.path.join(os.path.dirname(__file__), 'Ancient_Classifier.h5')
        return load_model(fname)



predictor = Predictor(200, 200)
predictor.init_test_data()
myModel = predictor.loadModel()
predictor.test_model(myModel)