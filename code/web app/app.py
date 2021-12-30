import os,glob
from flask import Flask,render_template,request,redirect, url_for,flash
import numpy as np
import cv2
from joblib import load
from PIL import Image
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model
import time
import functools
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from werkzeug.utils import secure_filename
import IPython.display
#from main2 import app
from flask import url_for

 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app=Flask(__name__)

tf.executing_eagerly()

def top3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
model1 = load_model('vggmodeldemo.h5',custom_objects={'top3_accuracy':top3_accuracy})

def load_img(img):
    max_dim = 512
    #img = Image.open(img)
    #height, width, channels = img.shape
    #img=cv2.imread(img)
    #width, height = img.size
    scale = 1024/max(img.size)
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
    img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
    img = np.expand_dims(img, axis=0)
    return img

def imshow(img, title=None):
  # Remove the batch dimension
    output = np.squeeze(img, axis=0)
  # Normalize for display 
    output = output.astype('uint8')
    plt.imshow(output)
    if title is not None:
        plt.title(title)
    plt.imshow(output)

def load_and_process_img(img):
    img = load_img(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(img):
    image = img.copy()
    if len(image.shape) == 4:
        image = np.squeeze(image, 0)
    assert len(image.shape) == 3, ("Input for image deprocessing must be an image of " "dimension [1, height, width, channel] or [height, width, channel]")
    if len(image.shape) != 3:
        raise ValueError("Invalid input for image deprocessing")
    image[:, :, 0] += 103.939
    image[:, :,1] += 116.779
    image[:, :, 2] += 123.68
    image = image[:, :,: :-1]
    image = np.clip(image, 0, 255).astype('uint8')
    return image

layers_content = ['block5_conv2'] 

# Style layer we are interested in
layers_style = ['block1_conv1','block2_conv1','block3_conv1', 'block4_conv1', 'block5_conv1']

num_layers_content = len(layers_content)
num_layers_styles = len(layers_style)





def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    model_outputs =  [vgg.get_layer(name).output for name in layers_style] +  [vgg.get_layer(name).output for name in layers_content]
    return models.Model(vgg.input, model_outputs)



def gm(ip):
    channels = int(ip.shape[-1])
    x = tf.reshape(ip, [-1, channels])
    n = tf.shape(x)[0]
    return  tf.matmul(x, x, transpose_a=True) / tf.cast(n, tf.float32)

def get_loss_content(content_base, trgt):
    return tf.reduce_mean(tf.square(trgt-content_base))

def get_loss_style(x, trgt):
    height, width, channels = x.get_shape().as_list()
    y = gm(x)
    return tf.reduce_mean(tf.square(y - trgt))


def get_features(m, path_content, path_style):
    content_img = load_and_process_img(path_content)
    style_img = load_and_process_img(path_style)
    outputs_styles = m(style_img)
    outputs_content = m(content_img)
    features_style = [layer_style[0] for layer_style in outputs_styles[:num_layers_styles]]
    features_content = [layer_content[0] for layer_content in outputs_content[num_layers_styles:]]
    return features_style, features_content




def compute_loss(model, weights_loss, img, gram_features, features_content):
    style_weight, content_weight = weights_loss
    op_model = model(img)
    style_output_features = op_model[:num_layers_styles]
    content_output_features = op_model[num_layers_styles:]
    style_score = 0
    content_score = 0
    weight_per_style_layer = 1.0 / float(num_layers_styles)
    for target_style, comb_style in zip(gram_features, style_output_features):
        style_score += weight_per_style_layer * get_loss_style(comb_style[0], target_style)

    weight_per_content_layer = 1.0 / float(num_layers_content)
    for target_content, comb_content in zip(features_content, content_output_features):
        content_score += weight_per_content_layer* get_loss_content(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score 
    return loss, style_score, content_score
  


def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        l = compute_loss(**cfg)
    total_loss = l[0]
    return tape.gradient(total_loss, cfg['init_image']), l




def style_transfer(content_path,style_path,num_iterations,content_weight=1e3,style_weight=1e-2):
    model = get_model() 
    for layer in model.layers:
        layer.trainable = False
    style_features, content_features = get_features(model, content_path, style_path)
    gram_style_features = [gm(style_feature) for style_feature in style_features]
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate= 1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')
    #iter_count = 1
    best_loss, best_img = float('inf'), None
    loss_weights = (style_weight, content_weight)
    cfg = {'model': model,'loss_weights': loss_weights,'init_image': init_image,'gram_style_features': gram_style_features,'content_features': content_features}
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means  
    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time() 
    
        if loss < best_loss: 
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % display_interval== 0:
            start_time = time.time()
            plot_img = init_image.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))        
            print('Total loss: {:.4e}, ' 'style loss: {:.4e}, ''content loss: {:.4e}, ''time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    #plt.switch_backend(plt.figure(figsize=(14,4)))
    #for i,img in enumerate(imgs):
     #   plt.subplot(num_rows,num_cols,i+1)
     #   plt.imshow(img)
      #  plt.xticks([])
       # plt.yticks([])
      
    return best_img, best_loss






@app.route('/')
def hello():
   return render_template('index.html')

@app.route('/detection',methods=['POST'])

def detect():
    file=request.files['image'] #Read image via file.stream
    img=Image.open(file.stream)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #img = cv2.imread('/content/17227.jpg')
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])

    probs = model1.predict(img)
    best_3 = np.argsort(probs, axis=1)[:,-3:]
    #print(best_3)
    predicts=model1.predict(img)
    names=[]
    accuracy=[]
    for i in best_3:
        for j in i:
            if(j==0):
                names.append("Albrecht Durer")
                accuracy.append(predicts[0][j]*100)
            elif(j==1):
                names.append("Alfred Sisley")
                accuracy.append(predicts[0][j]*100)
            elif(j==2):
                names.append("Boris Kustodiev")
                accuracy.append(predicts[0][j]*100)
            elif(j==3):
                names.append("Camille Corot")
                accuracy.append(predicts[0][j]*100)
            elif(j==4):
                names.append("Camille Pissarro")
                accuracy.append(predicts[0][j]*100)
            elif(j==5):
                names.append("Childe Hassam")
                accuracy.append(predicts[0][j]*100)
            elif(j==6):
                names.append("Claude Monet")
                accuracy.append(predicts[0][j]*100)
            elif(j==7):
                names.append("Edgar Degas")
                accuracy.append(predicts[0][j]*100)
            elif(j==8):
                names.append("Eugene Boudin")
                accuracy.append(predicts[0][j]*100)
            elif(j==9):
                names.append("Eyvind Earle")
                accuracy.append(predicts[0][j]*100)
            elif(j==10):
                names.append("Fernand Leger")
                accuracy.append(predicts[0][j]*100)
            elif(j==11):
                names.append("Giovanni Battista Piranesi")
                accuracy.append(predicts[0][j]*100)
            elif(j==12):
                names.append("Gustave Dore")
                accuracy.append(predicts[0][j]*100)
            elif(j==13):
                names.append("Henri Martin")
                accuracy.append(predicts[0][j]*100)
            elif(j==14):
                names.append("Henri Matisse")
                accuracy.append(predicts[0][j]*100)
            elif(j==15):
                names.append("Ilya Repin")
                accuracy.append(predicts[0][j]*100)
            elif(j==16):
                names.append("Isaac Levitan")
                accuracy.append(predicts[0][j]*100)
            elif(j==17):
                names.append("Ivan Aivazovsky")
                accuracy.append(predicts[0][j]*100)
            elif(j==18):
                names.append("Ivan Shishkin")
                accuracy.append(predicts[0][j]*100)
            elif(j==19):
                names.append("James Tissot")
                accuracy.append(predicts[0][j]*100)
            elif(j==20):
                names.append("John Singer Sargent")
                accuracy.append(predicts[0][j]*100)
            elif(j==21):
                names.append("Marc Chagall")
                accuracy.append(predicts[0][j]*100)
            elif(j==22):
                names.append("Martiros Saryan")
                accuracy.append(predicts[0][j]*100)
            elif(j==23):
                names.append("Nicholas Roerich")
                accuracy.append(predicts[0][j]*100)
            elif(j==24):
                names.append("Odilon Redon")
                accuracy.append(predicts[0][j]*100)
            elif(j==25):
                names.append("Pablo Picasso")
                accuracy.append(predicts[0][j]*100)
            elif(j==26):
                names.append("Paul Cezanne")
                accuracy.append(predicts[0][j]*100)
            elif(j==27):
                names.append("Paul Gauguin")
                accuracy.append(predicts[0][j]*100)
            elif(j==28):
                names.append("Pierre-Auguste Renoir")
                accuracy.append(predicts[0][j]*100)
            elif(j==29):
                names.append("Pyotr Konchalovsky")
                accuracy.append(predicts[0][j]*100)
            elif(j==30):
                names.append("Raphael Kirchner")
                accuracy.append(predicts[0][j]*100)
            elif(j==31):
                names.append("Rembrandt")
                accuracy.append(predicts[0][j]*100)
            elif(j==32):
                names.append("Salvador Dali")
                accuracy.append(predicts[0][j]*100)
            elif(j==33):
                names.append("Theophile Steinlen")
                accuracy.append(predicts[0][j]*100)
            elif(j==34):
                names.append("Vincent van Gogh")
                accuracy.append(predicts[0][j]*100)
            elif(j==35):
                names.append("Zdislav Beksinski")
                accuracy.append(predicts[0][j]*100)
            elif(j==36):
                names.append("Zinaida Serebriakova")
                accuracy.append(predicts[0][j]*100)
    names.reverse()
    accuracy.reverse()
    print(names)
    print(accuracy)
    if(accuracy[0]>50):
        print("Highly Similar")
    elif(accuracy[0]<=50 and accuracy[0]>=45):
        print("Somewhat Similar")
    else:
        print("Not Similar")
    #print(img)
    return 'Hi'
    #return 'Hello'

@app.route('/conversion',methods=['POST'])
def convert():
    

     
    file=request.files['image'] #Read image via file.stream
    img1=Image.open(file.stream)
    #img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    file=request.files['image'] #Read image via file.stream
    img2=Image.open(file.stream)
    #img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    

    best, best_loss = style_transfer(img1 ,img2 , num_iterations=50)
    print(img1)
    print(img2)
    print(Image.fromarray(best))
    im = Image.fromarray(best)
    im.save('test.jpeg')
    im.show()


with app.test_request_context('/'):
    url_for('hello')

#with app.test_request_context('/conversion'):
 #   url_for('convert')

#with app.test_request_context('/detection'):
 #   url_for('detect')

if __name__ == "__main__":
    app.run(debug=True)
#app.run()