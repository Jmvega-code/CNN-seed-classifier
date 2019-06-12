# Installation of the modules and packages
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from IPython.display import display
from PIL import Image
from sklearn.metrics import confusion_matrix,classification_report

# dimensions of our images.
img_size = 150
# Creating the train and validation directories
train_data_dir = './data/train'
validation_data_dir = './data/validation'
nb_train_samples = 33 * 5
nb_validation_samples = 17 * 5
epochs = 100
batch_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_size, img_size)
else:
    input_shape = (img_size, img_size, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size = 3, input_shape=input_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation = 'softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



# With this for loop we can see a sample of generated images based in a single picture
#img = load_img('/Users/jschrodinger/Desktop/TFM/data/train/daniel/daniel00002.jpg')  # this is a PIL image
#img.size
#x = img_to_array(img)  # this is a Numpy array with shape (3, 500, 500)
#x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 500, 500)

# the .flow() command below generates batches of randomly transformed images and saves the results to the `preview/` directory
#i = 0
#for batch in train_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='daniel', save_format='jpeg'):
#    i += 1
#    if i > 5:
#        break  # otherwise the generator would loop indefinitely
# CALLBACKS
NAME = "CNN-seed-classifier_rmsprop"
# with tensorboard we can visualize the loss and accuracy values in graph
#  typing on console 'tensorboard --logdir=logs/'
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# this reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# ModelCheckpoint saves the model after each epoch if the validation loss improves
checkpointer = ModelCheckpoint(filepath='./tmp/model_rmsprop.hdf5', verbose=1, save_best_only=True)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_size, img_size),
    batch_size = batch_size,
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_size, img_size),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle=False)

model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,
    callbacks = [tensorboard, reduce_lr, checkpointer],
    verbose = 2)
# steps_per_epoch = 33*5 (165)  // 3 = 55 
# validation_steps = 17*5 (85) // 3 = 28.3 = 29

# Performance Measurements
# Confusion matrix is created with the best model saved by Checkpoints
model = tf.keras.models.load_model('/Users/jschrodinger/Desktop/TFM/tmp/model_rmsprop.hdf5')

predictions = model.predict_generator(validation_generator,  nb_validation_samples // batch_size + 1)

y_pred = np.argmax(predictions, axis=1)

true_classes = validation_generator.class_indices

class_labels = list(validation_generator.class_indices.keys())   
print('\nConfusion Matrix\n')
print(class_labels)

print(confusion_matrix(validation_generator.classes, y_pred))

# Confusion Metrics: Accuracy, Precision, Recall & F1 Score
report = classification_report(validation_generator.classes, y_pred, target_names= class_labels)
print('\nClassification Report\n')
print(report)

# Plotting the model architecture and saving it in a .png file
plot_model(model, to_file='model_rmsprop_plot.png', show_shapes=True, show_layer_names=True)

# Printing the model Summary
print(model.summary())