import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def get_conf_mat(y_pred, y_target, n_cats):
    """Build confusion matrix from scratch.
    (This part could be a good student assignment.)
    """
    conf_mat = np.zeros((n_cats, n_cats))
    n_samples = y_target.shape[0]
    for i in range(n_samples):
        _t = y_target[i]
        _p = y_pred[i]
        conf_mat[_t, _p] += 1
    norm = np.sum(conf_mat, axis=1, keepdims=True)
    return conf_mat / norm


def vis_conf_mat(conf_mat, cat_names, acc):
    """Visualize the confusion matrix and save the figure to disk."""
    n_cats = conf_mat.shape[0]

    fig, ax = plt.subplots()
    # figsize=(10, 10)

    cmap = cm.Blues
    im = ax.matshow(conf_mat, cmap=cmap)
    im.set_clim(0, 1)
    ax.set_xlim(-0.5, n_cats - 0.5)
    ax.set_ylim(-0.5, n_cats - 0.5)
    ax.set_xticks(np.arange(n_cats))
    ax.set_yticks(np.arange(n_cats))
    ax.set_xticklabels(cat_names)
    ax.set_yticklabels(cat_names)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    for i in range(n_cats):
        for j in range(n_cats):
            text = ax.text(j, i, round(
                conf_mat[i, j], 2), ha="center", va="center", color="w")

    cbar = fig.colorbar(im)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    _title = 'Normalized confusion matrix, acc={0:.2f}'.format(acc)
    ax.set_title(_title)

    # plt.show()
    _filename = 'conf_mat.png'
    plt.savefig(_filename, bbox_inches='tight')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
FINETUNE_EPOCHS = 10

# CHANGE THIS FOR TASK 1 vs TASK 2
DATASET_PATH = r"C:\Users\Matthew Walters\Documents\VS_Code\Robot Vision Class\Classifier\8FishSpecies\8 Fish Species"   # or "OysterShell"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# ---------------------------------------------------------
# DATA GENERATORS (with augmentation)
# ---------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_gen.num_classes

# ---------------------------------------------------------
# MODEL: TRANSFER LEARNING (ResNet50)
# ---------------------------------------------------------
model = Sequential([ 
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)), 
    BatchNormalization(), 
    MaxPooling2D(), 
    Conv2D(64, (3,3), activation='relu', padding='same'), 
    BatchNormalization(), 
    MaxPooling2D(), 
    Conv2D(128, (3,3), activation='relu', padding='same'), 
    BatchNormalization(), 
    MaxPooling2D(), 
    Conv2D(256, (3,3), activation='relu', padding='same'), 
    BatchNormalization(), 
    MaxPooling2D(), 
    Flatten(), 
    Dense(256, activation='relu'), 
    Dropout(0.4), 
    Dense(num_classes, activation='softmax') ])

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------------
# TRAINING (Stage 1: frozen backbone)
# ---------------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)


# ---------------------------------------------------------
# EVALUATE ON TEST SET
# ---------------------------------------------------------
test_loss, test_acc = model.evaluate(test_gen)
print("Test accuracy:", test_acc)

# ---------------------------------------------------------
# PREDICTIONS FOR CONFUSION MATRIX
# ---------------------------------------------------------
y_true = test_gen.classes

pred_probs = model.predict(test_gen)
y_pred = np.argmax(pred_probs, axis=1)

class_names = list(test_gen.class_indices.keys())

# ---------------------------------------------------------
# CONFUSION MATRIX (using instructor's functions)
# ---------------------------------------------------------
# These functions come from the LBPSVM sample code
# Make sure you import them at the top of your file:
# from lbpsvm_utils import get_conf_mat, vis_conf_mat

conf_mat = get_conf_mat(y_true, y_pred, len(class_names))
vis_conf_mat(conf_mat, class_names, f"{DATASET_PATH}_confusion_matrix.png")

print("Confusion matrix saved.")
