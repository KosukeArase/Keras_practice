# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import PIL

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions



filename = sys.argv[1]

model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
# include_top: 全結合層を含めるか，input_tensor: 自分でモデルに画像を入力したいときに使う？Fine-tuning のときに利用，input_shape: include_top=True なのでデフォでいい

img = image.load_img(filename, target_size=(224, 224)) # 自動的にリサイズ
# img.show()

x = image.img_to_array(img) # PIL (Python Imaging Library) 形式の画像を NumPy array に変換


x = np.expand_dims(x, axis=0) # 3次元テンソル (rows, cols, channels) を4次元テンソル (samples, rows, cols, channels) に変換

preds = model.predict(preprocess_input(x))
results = decode_predictions(preds, top=5)[0]

for result in results:
    print(result)
