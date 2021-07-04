
import os
import random
import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import load_model
from keras.utils import np_utils
from matplotlib import pyplot as plt


model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
#在model內新增捲積層，其中input的大小為(28,28)，捲積核大小為3x3。激活函數為 relu函數Rectified Linear Unit
model.add(MaxPooling2D(pool_size=(2,2)))
#模型內建立池化層
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
#在模型內建立多層捲積層
model.add(MaxPooling2D(pool_size=(2,2)))
#模型內建立池化層
model.add(Dropout(0.25))

model.add(Flatten())
#加入Flatten層
model.add(Dropout(0.1))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units =10,activation='softmax'))

model.summary()

def data_x_y_preprocess(datapath):

    img_row,img_col= 28,28
    data_x = np.zeros((28,28)).reshape(1,28,28)
    pic_count = 0
    data_y = []
    num_class=10

    for root, dirs, files in os.walk(datapath):
        for f in files:
            
            label = int(root.split("\\")[2])
            data_y.append(label)
            fullpath= os.path.join(root,f)
            #print(os.path.join(root,f))
            img = Image.open(fullpath)
          
            img = (np.array(img)/255).reshape(1,28,28)
            data_x = np.vstack((data_x,img))
            pic_count+=1
           
             
    data_x= np.delete(data_x,[0],0)    
    data_x = data_x.reshape(pic_count,img_row,img_col,1)
    data_y = np_utils.to_categorical(data_y,num_class)
    return data_x,data_y


data_train_X,data_train_Y= data_x_y_preprocess("train_image")
data_test_X,data_test_Y= data_x_y_preprocess("test_image")


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #編譯模型

#print(data_train_X,data_train_Y)
#print(data_test_X,data_test_Y)
train_history = model.fit(data_train_X,data_train_Y,batch_size=32,epochs=150,verbose=1,validation_split=0.1)
score = model.evaluate(data_test_X, data_test_Y,verbose=0)

model.save_weights('./save_weight')

print('Test loss',score[0])
print('Test accuracy',score[1])
