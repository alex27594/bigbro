from keras.models import Model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from scipy.ndimage.filters import gaussian_filter

data_root = "/home/alexander/PycharmProjects/car_detection/brand_image/"
train_path = data_root + "train/"
test_path = data_root + "test/"

print(train_path)

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.5,
    fill_mode='nearest',
    preprocessing_function=lambda item: gaussian_filter(item, sigma=3)
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.5,
    fill_mode='nearest',
    preprocessing_function=lambda item: gaussian_filter(item, sigma=3)
)

batch_size = 100

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

filepath="saved_steps/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', period=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=5e-3, patience=5, mode='auto')
callbacks_list = [checkpoint, early_stopping]

model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=3,
    callbacks=callbacks_list
)
