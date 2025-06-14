import tensorflow as tf
from keras import layers
from sklearn.utils import shuffle
from tensorflow.python.data import AUTOTUNE

training_set=tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\Ishan Khan\PycharmProjects\PneumoniaDetection\chest_xray\train",
    labels='inferred',
    label_mode="binary",
    shuffle=True,
    batch_size=32,
    image_size=(150,150)
)

test_set=tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\Ishan Khan\PycharmProjects\PneumoniaDetection\chest_xray\test",
    labels='inferred',
    label_mode="binary",
    shuffle=False,
    batch_size=32,
    image_size=(150,150)
)

data_aug=tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("Horizontal"),
    layers.RandomZoom(0.2),
    layers.RandomRotation(0.1)
])

norm_layer=layers.Rescaling(1./255)

training_set=training_set.map(lambda x,y:(data_aug(x),y))
test_set=test_set.map(lambda x,y:(norm_layer(x),y))

AUTOTUNE=tf.data.AUTOTUNE
training_set = training_set.prefetch(buffer_size=AUTOTUNE)
test_set = test_set.prefetch(buffer_size=AUTOTUNE)


from tensorflow.keras.layers import Flatten, Dense, MaxPool2D,Conv2D

cnn=tf.keras.Sequential([
    Conv2D(filters=64,kernel_size=3,activation="relu",input_shape=(150,150,3)),
    MaxPool2D(pool_size=2,strides=2),
    Conv2D(filters=128,kernel_size=3,activation="relu"),
    MaxPool2D(pool_size=2,strides=2),
    Flatten(),
    Dense(units=256,activation="relu"),
    Dense(units=1,activation="sigmoid")
])

cnn.compile(optimizer="adam",
             loss='binary_crossentropy',
             metrics=['accuracy'])

cnn.fit(training_set, epochs=25, validation_data=test_set)


cnn.save("pneumonia_model.h5")


