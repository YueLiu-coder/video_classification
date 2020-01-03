from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from UCFdata import DataSet
import keras
import os
import warnings
warnings.simplefilter('ignore')


data = DataSet()

print("xxx")
# Helper: Save the min val_loss model in each epoch.
checkpointer = ModelCheckpoint(
    filepath='./data/checkpoints/inception_diff.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
# patience: number of epochs with no improvement after which training will be stopped.
early_stopper = EarlyStopping(patience=100)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir='./data/logs/')

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        './data/test_diff/',
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        './data/test_diff/',
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')

    return train_generator, validation_generator

def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a softmax layer
    predictions = Dense(len(data.classes), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def acc_top5(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

def fine_tune_inception_layer(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', acc_top5])

    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

def main(weights_file):

    model = get_model()
    generators = get_generators()

    if weights_file is None:
        print("Training Top layers.")
        model = train_model(model, 10, generators)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    # model = fine_tune_inception_layer(model)
    model = train_model(model, 1000, generators,
                        [checkpointer, early_stopper, tensorboard])

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    weights_file = '/data2/public/ly/UCF-101_video_classification/data/checkpoints/inception_diff.088-3.06.hdf5'
    main(weights_file)