import tensorflow as tf
import keras_preprocessing.image as process_im
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications import vgg19
from keras.models import Model
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import functools
import IPython.display
import numpy as np
import cv2


def load_file(image_path):
    image = Image.open(image_path)
    max_dim = 512
    factor = max_dim / max(image.size) # resize rate
    '''
    Image.NEAREST ：低质量
    Image.BILINEAR：双线性
    Image.BICUBIC ：三次样条插值
    Image.ANTIALIAS：高质量
    '''
    image = image.resize((round(image.size[0] * factor), round(image.size[1] * factor)), Image.ANTIALIAS)
    im_array = process_im.img_to_array(image) #to array
    im_array = np.expand_dims(im_array, axis=0)  # adding extra axis to the array as to generate a
    # batch of single image

    return im_array

def show_im(img):
    img=np.squeeze(img,axis=0) #squeeze array to drop batch axis #降一維
    return np.uint8(img)

content_path = 'leo.png'
style_path = 'plant1.png'
plt.figure(figsize=(10,10))
content = load_file(content_path)
style = load_file(style_path)
# plt.subplot(1,2,1)
# show_im(content,'Content Image')
# plt.subplot(1,2,2)
# show_im(style,'Style Image')
# plt.show()

def img_preprocess(img_path):
    image=load_file(img_path)
    img=tf.keras.applications.vgg19.preprocess_input(image) #to array
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0) #down 1 ndim
    assert len(x.shape) == 3  # assert 用於判斷一個表達式，若無滿足該表達式的條件，則直接觸發異常狀態，而不會接續執行後續的程式碼

    # 設定RGB顏色的中心點 (Remove zero-center by mean pixel)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 'BGR'->'RGB'
    x = x[:, :, ::-1]  # converting BGR to RGB channel
    x = np.clip(x, 0, 255).astype('uint8') #min 0 max 255

    return x

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
number_content=len(content_layers)
number_style =len(style_layers)


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_output = [vgg.get_layer(layer).output for layer in content_layers]
    style_output = [vgg.get_layer(layer).output for layer in style_layers]
    model_output = style_output + content_output
    return models.Model(vgg.input, model_output)

def get_content_loss(noise,target):
    loss = tf.reduce_mean(tf.square(noise-target))
    return loss

# 計算 風格 loss 的 gram matrix
def gram_matrix(tensor):
    channels=int(tensor.shape[-1])
    vector=tf.reshape(tensor,[-1,channels]) #x,[-1,channels] -> [[1,2,3],[4,5,6]]
    n=tf.shape(vector)[0]
    gram_matrix=tf.matmul(vector,vector,transpose_a=True)
    '''
    https://blog.csdn.net/qq_37591637/article/details/103473179
    矩正相乘
    [1 2 3 * [0 0 1   = [1*0+2*1+3*3 , 1*0+2*3+3*3,..
    4 5 6]    1 3 2      4*0+5*1+6*3 ....              ]
              3 3 4] 
    '''
    return gram_matrix/tf.cast(n,tf.float32) #型別轉換

def get_style_loss(noise,target):
    gram_noise=gram_matrix(noise)
    #gram_target=gram_matrix(target)
    loss=tf.reduce_mean(tf.square(target-gram_noise)) #計算reduce_mean所有這些浮點數的平均值。
    return loss


def get_features(model, content_path, style_path):
    content_img = img_preprocess(content_path)
    style_image = img_preprocess(style_path)

    content_output = model(content_img)
    style_output = model(style_image)

    content_feature = [layer[0] for layer in content_output[number_style:]]
    style_feature = [layer[0] for layer in style_output[:number_style]]
    return content_feature, style_feature


def compute_loss(model, loss_weights, image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights  # style weight and content weight are user given parameters
    # that define what percentage of content and/or style will be preserved in the generated image

    output = model(image)
    content_loss = 0
    style_loss = 0

    noise_style_features = output[:number_style]
    noise_content_feature = output[number_style:]

    weight_per_layer = 1.0 / float(number_style)
    for a, b in zip(gram_style_features, noise_style_features):
        style_loss += weight_per_layer * get_style_loss(b[0], a)

    weight_per_layer = 1.0 / float(number_content)
    for a, b in zip(noise_content_feature, content_features):
        content_loss += weight_per_layer * get_content_loss(a[0], b)

    style_loss *= style_weight
    content_loss *= content_weight

    total_loss = content_loss + style_loss

    return total_loss, style_loss, content_loss


def compute_grads(dictionary):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**dictionary)

    total_loss = all_loss[0]
    return tape.gradient(total_loss, dictionary['image']), all_loss


def run_style_transfer(content_path, style_path, epochs=500, content_weight=1e3, style_weight=1e-2):
    model = get_model()

    for layer in model.layers:
        layer.trainable = False

    content_feature, style_feature = get_features(model, content_path, style_path)
    style_gram_matrix = [gram_matrix(feature) for feature in style_feature]

    noise = img_preprocess(content_path)
    noise = tf.Variable(noise, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1) #模糊因子

    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    dictionary = {'model': model,
                  'loss_weights': loss_weights,
                  'image': noise,
                  'gram_style_features': style_gram_matrix,
                  'content_features': content_feature}

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    '''
    即compute_gradients和apply_gradients，
    前者用於計算梯度，
    後者用於使用計算得到的梯度來更新對應的variable。下面對這兩個函數做具體介紹。
    '''
    for i in range(epochs):
        grad, all_loss = compute_grads(dictionary)
        total_loss, style_loss, content_loss = all_loss
        optimizer.apply_gradients([(grad, noise)])
        '''
         tf.clip_by_value的用法tf.clip_by_value(A, min, max)：
         輸入一個張量A，把A中的每一個元素的值都壓縮在min和max之間。
         小於min的讓它等於min，大於max的元素的值等於max
        '''
        clipped = tf.clip_by_value(noise, min_vals, max_vals)
        '''
        https://medium.com/ai-blog-tw/tensorflow-%E4%BB%80%E9%BA%BC%E6%98%AFassign-operator-%E4%BB%A5tf-assign%E5%AF%A6%E4%BD%9Ccounter-184479257531
        assign 剛剛上面的寫法問題在於，不管你跑幾次，
        var這個tf.Variable的值都從未更新過，所以需要使用tf.assign指定新的數值
        '''
        noise.assign(clipped)

        if total_loss < best_loss:
            best_loss = total_loss
            best_img = deprocess_img(noise.numpy())

        # for visualization

        if i % 5 == 0:
            plot_img = noise.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Epoch: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '.format(total_loss, style_loss, content_loss))

    IPython.display.clear_output(wait=True)

    return best_img, best_loss, imgs

best, best_loss,image = run_style_transfer(content_path,
                                     style_path, epochs=500)

img = cv2.imread(content_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img[:,262:,:] = best[:,262:,:]

content = show_im(content)
style = show_im(style)

result = [content,style,best,img]
result_lable = ['origan','style','remix','half_mix']
for i in range(len(result)):
    plt.subplot(2,2,i+1)
    plt.title(result_lable[i])
    plt.xticks([]);plt.yticks([])
    plt.imshow(result[i])

plt.savefig('four_step_style.png')
plt.show()

# plt.subplot(1,3,3)
# plt.imshow(best)
# show_im(style,'Style Image')

# plt.xticks([]);plt.yticks([])
# plt.savefig('leo_transfer.png')
#
# img = cv2.imread('leo.png')
# style = cv2.imread('leo_transfer.png')
#
# edge = 262
# img_right = img[:,edge:,:]
# img[:,edge:,:] = style
#
# cv2.imshow('result_finish',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
