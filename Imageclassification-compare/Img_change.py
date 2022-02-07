
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D,MaxPool2D,AvgPool2D\
    ,Flatten


rootpath='G:/traffic_data/Training'
images=[]
labels=[]
def load_data(rootpath):
    for d in os.listdir(rootpath):

        label_path = rootpath + '/' + d
        print(label_path)
        if(os.path.isdir(label_path)):
            print(label_path)
            for i in os.listdir(label_path):
                image_path=label_path+'/'+i
                if(i.endswith(".ppm")):
                    image_load=Image.open(image_path).resize((64,64),Image.BILINEAR)
                    image_load = np.asarray(image_load)
            #如果不是三通道 则其他GB层生成0
                    images.append(image_load)
                    labels.append(int(d))
    return images, labels

images,labels=load_data(rootpath)

images=np.asarray(images)
labels=np.asarray(labels)

print(images.shape)

print(labels.shape)

plt.imshow(images[20])
plt.show()
print(labels)




model=keras.models.Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5),activation='relu',input_shape=(64,64,3)))
model.add(MaxPool2D(2,2))

model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Flatten())

model.add(Dense(1024,activation='relu'))

model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(62, activation='softmax'))

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])

#print(tf.__version__)
#images = keras.backend.cast_to_floatx(images)
#label2=keras.backend.cast_to_floatx(label2)


model.fit(images,labels,batch_size=20,epochs=10)

model.summary()


#print(dataload.images)
#print(dataload.label2)
print(tf.__version__)





