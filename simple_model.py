import keras
import tensorflow as tf
import numpy as np
import os

from skimage.io import imread,imsave
from skimage.color import rgb2gray,gray2rgb,rgb2lab,lab2rgb
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.preprocessing.image import img_to_array,load_img
from keras.preprocessing.image import ImageDataGenerator

# 将RGB彩色图片转换为LAB制式，并将L作为x,AB作为y
def getGrayImg(img_file):
    image = img_to_array(load_img(img_file))
    image_shape = image.shape
    image = np.array(image, dtype=float)
    # 除以255进行归一化后，切片读取各通道，将L赋值给x,AB层为y
    x = rgb2lab(1.0 / 255 * image)[:, :, 0]
    y = rgb2lab(1.0 / 255 * image)[:, :, 1:]
    # 将像素点的值归一化
    y /= 128
    # 转成batch，将三维图片转换为神经网络默认的四维输入
    x = x.reshape(1, image_shape[0], image_shape[1], 1)
    y = y.reshape(1, image_shape[0], image_shape[1], 2)

    return x, y, image_shape

# 建立模型
def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    # 优化器
    model.compile(optimizer='rmsprop',loss='mse')

    return  model

# 训练模型
def train():
    img_path = './img_2.jpeg'
    x,y,img_shape = getGrayImg(img_path)

    print('img_shape:',img_shape)
    model = build_model()
    print(model)
    num_epochs = 6000
    batch_size = 6
    model_file = 'simple_model1.h5'
    # 给定输入输出x和y,计算权重
    model.fit(x,y,batch_size=1,epochs=100)
    model.save(model_file)
# train()

# 预测模型（着色）输入为彩色图片
def preColor():
    img_path = './img_1.jpeg'
    x,y,img_shape = getGrayImg(img_path)
    model = build_model()
    model.load_weights = 'simple_model2.h5'
    print('model loaded..',img_shape)
    y = model.predict(x)
    y *= 128
    print('y:',y)
    tmp = np.zeros((img_shape[0], img_shape[1], 3))
    # x为
    tmp[:, :, 0] = x[0][:, :, 0]
    # y为ab
    tmp[:, :, 1:] = y[0]
    # LAB转换为RGB
    imsave('result_1.png',lab2rgb(tmp))
    # 灰度图
    imsave('gray_1.png',rgb2gray(lab2rgb(tmp)))
# preColor()

# 输入为灰度图
def pre_process_image_for_colorize(img_file, target_size=(256, 256)):
    if os.path.exists(img_file):
        img = np.asarray(imread(img_file), dtype=float)
        # 指定大小
        img_shape = img.shape
        assert target_size[0] <= img_shape[0] and target_size[1] <= img_shape[1], 'image file must bigger than ' \
                                                                                  'target_size'
        crop_w = np.random.randint(0, img.shape[0] - target_size[0] + 1)
        crop_h = np.random.randint(0, img.shape[1] - target_size[1] + 1)
        img_random_cropped = img[crop_w:crop_w + target_size[0], crop_h:crop_h + target_size[1]]
        img_rgb = gray2rgb(img_random_cropped)
        # 转为LAB前归一化处理
        img_lab = rgb2lab(img_rgb / 225.)
        x = img_lab[:, :, 0]
        x = np.reshape(x, (target_size[0], target_size[1], 1))
        return x
      
# 输入为黑白图片的文件夹
def colorize_gray(test_img_dir):
    model = build_model()
    model.load_weights('simple_model.h5')
    print('model loaded..')
    target_size = (256, 256)

    all_test_img_files = [os.path.join(test_img_dir, i) for i in os.listdir(test_img_dir) if i.endswith('png')]
    for img_file in all_test_img_files:
        f_name = os.path.basename(img_file).split('.')[0]
        x_origin = pre_process_image_for_colorize(img_file=img_file)
        x = np.expand_dims(x_origin, axis=0)

        y = model.predict(x)
        y *= 128
        # L为x AB为y
        tmp = np.zeros((target_size[0], target_size[1], 3))
        tmp[:, :, 0] = x_origin[:, :, 0]
        tmp[:, :, 1:] = y[0]
        # 将LAB图转为RGB图
        imsave(os.path.join(test_img_dir, 'result_simple_{}.jpg'.format(f_name)), lab2rgb(tmp))
        print('result image of {} saved.'.format(img_file))


# colorize_gray('/Users/feishuoren/Projects/color_img/data')
