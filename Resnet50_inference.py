import numpy as np 
import matplotlib.pyplot as plt
import os 
from PIL import Image
os.environ['GLOG_minloglevel'] = '2' # Set so that Caffe stops printing so much stuff to console
import caffe 
os.environ['GLOG_minloglevel'] = '0'
from labels import labels 

caffe.set_mode_cpu()
# load the network 
net = caffe.Net('./models/ResNet-50-deploy.prototxt','./models/ResNet-50-model.caffemodel', caffe.TEST)

def preprocess_image(image):
    # set up preprocessing image steps
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([104,117,123]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    # Load and resize images 
    im = caffe.io.load_image(image)
    im = caffe.io.resize_image(im, (224,224,3))
    net.blobs['data'].reshape(1,3,224,224)
    net.blobs['data'].data[:,:,:] = transformer.preprocess('data', im)
    


def run_inference(image): # Performs one forward pass of Resnet50 with an image 

    im = preprocess_image(image)
    out = net.forward()
    prediction = out['prob'][0].argmax(axis=0) 

    print ("Image label {}, Class - {}".format(prediction, labels[prediction]))
    return prediction

if __name__ == "__main__":
    run_inference('./elephant.jpg')    