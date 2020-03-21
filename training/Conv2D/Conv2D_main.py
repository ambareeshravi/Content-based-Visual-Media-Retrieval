import os, cv2, numpy as np
from glob import glob
from random import shuffle
from tqdm import tqdm
import pickle as pkl

from keras.models import Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.datasets import cifar10

from utils import *

try:
    # to run on colab
    from google.colab import drive
    def mount_drive():
        drive.mount('/content/drive/')
    mount_drive()
    print("Runnning on GOOGLE COLAB")
    isCloud = True
except:
    print("Runnning on LOCAL SYSTEM")
    isCloud = False

class Conv2DTrainer:
    def __init__(self):
        self.epochs = 50
        self.batch_size = 32
        self.validation_split = 0.15
        
        self.load_CIFAR_data()
        self.load_models()
        self.model_save_path = "./"
        if isCloud: self.model_save_path = "/content/drive/My Drive/"

    def get_bw(self, clr_img):
        return np.expand_dims(cv2.cvtColor(clr_img, cv2.COLOR_BGR2GRAY), axis=-1)

    def get_bw_data(self, x_train):
        return np.array([self.get_bw(i) for i in x_train])
        
    # load data
    def load_CIFAR_data(self,):
        self.classes_id = {'airplane' : 0, 'automobile' : 1, 'bird' : 2, 'cat' : 3, 'deer' : 4, 'dog' : 5, 'frog' : 6, 'horse' : 7, 'ship' : 8, 'truck' : 9}
        (x_train, self.y_train), (x_test, self.y_test) = cifar10.load_data()

        # normalization
        self.x_train = x_train.astype('float32') 
        self.x_test = x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        self.x_train_g, self.x_test_g = self.get_bw_data(x_train), self.get_bw_data(x_test)

    # load models
    def load_models(self):
        self.mnv2_clr = MobileNetV2(input_shape = (32,32,3), weights = None, classes=10)
        self.mnv2_bw = MobileNetV2(input_shape = (32,32,1), weights = None, classes=10)

    def train_model(self, model_fn = "grey"):
        if model_fn =='grey':  
            model, x_train, y_train = self.mnv2_bw, self.x_train_g, self.y_train 
        else:
            model, x_train, y_train = self.mnv2_clr, self.x_train, self.y_train
        print("Training", model_fn,"model")
        
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, shuffle=True, verbose=1, validation_split = self.validation_split)
        
        model.save(self.model_save_path + "CIFAR_" +model_fn+ ".h5")
    
    def train_models(self, ):
        # train grey model
        self.train_model('grey')
        # train color model
        self.train_model('color')
    
class Conv2DTester(Conv2DTrainer):
    def __init__(self, color_model_path = 'CIFAR_color.h5', grey_model_path = 'CIFAR_grey.h5'):
        Conv2DTrainer.__init__(self)
        self.clr_model = load_model(self.model_save_path + color_model_path)
        self.gs_model = load_model(self.model_save_path + grey_model_path)
        self.clr_snip = snip_model(self.clr_model, "global_average_pooling2d_1")
        self.gs_snip = snip_model(self.gs_model, "global_average_pooling2d_1")
        
    def predict_image(self, model, fl, gray = False, read = False, resize = False):
        if read: fl = cv2.imread(fl)
        if gray: fl = self.get_bw(fl)
        if resize: fl = resize_image(fl)
        fl = fl/255
        fl = np.expand_dims(fl, axis = 0)
        pred = model.predict(fl)
        return np.squeeze(pred)
    
    def get_class(self, file_name):
        for k, v in class_id.items():
            if v in file_name:
                return k

    def test_models(self, save_file = None):
        Xt = self.x_test
        yt = self.y_test
        cm = self.clr_model
        cms = self.clr_snip
        gm = self.gs_model
        gms = self.gs_snip

        test_set = [(x,y) for x, y in zip(Xt,yt)]

        results_dict = dict()
        for idx, (image_array, label) in tqdm(enumerate(test_set)):
            results_dict[idx] = {
                "img_array": image_array,
                "actual_label": np.squeeze(label),

                "clr_label": np.argmax(self.predict_image(cm, image_array, gray=False)),
                "clr_features": self.predict_image(cms, image_array, gray=False),

                "gs_label": np.argmax(self.predict_image(gm, image_array, gray=True)),
                "gs_features": self.predict_image(gms, image_array, gray=True)
            }
            
        if save_file != None:
            save_pickle(self.model_save_path + save_file + ".pkl", results_dict)

        return results_dict
    
    def sample_testing(self, complete_results, save_file = None, n_samples = 100):
        random_indices = np.random.random_integers(0, len(complete_results), n_samples)
        sample_test_set = dict()
        for idx, i in enumerate(random_indices):
            sample_test_set[idx] = complete_results[i]
        
        if save_file!=None:
            save_pickle(self.model_save_path + save_file + ".pkl", sample_test_set)
        
if __name__ == '__main__':
    trainer = Conv2DTrainer()
    trainer.train_models()
    
    tester = Conv2DTester()
    complete_results = tester.test_models("complete_results")
    sampled_results = tester.sample_testing(complete_results, "sampled_results")
    