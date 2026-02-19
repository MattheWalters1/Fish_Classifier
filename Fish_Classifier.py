import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 80
fish = True

DATASET_PATH = "/content/drive/MyDrive/DataSets/Oyster Shell"
if fish:
    DATASET_PATH = "/content/drive/MyDrive/DataSets/8 Fish Species"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# ---------------------------------------------------------
# CONFUSION MATRIX HELPERS
# ---------------------------------------------------------
def get_conf_mat(y_true, y_pred, n_cats):
    conf_mat = np.zeros((n_cats, n_cats))
    n_samples = y_true.shape[0]
    for i in range(n_samples):
        t = y_true[i]
        p = y_pred[i]
        conf_mat[t, p] += 1
    norm = np.sum(conf_mat, axis=1, keepdims=True)
    return conf_mat / norm

def vis_conf_mat(conf_mat, cat_names, acc):
    n_cats = conf_mat.shape[0]
    fig, ax = plt.subplots()

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
            ax.text(j, i, round(conf_mat[i, j], 2),
                    ha="center", va="center", color="w")

    fig.colorbar(im)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Normalized confusion matrix, acc={acc:.2f}')

    plt.savefig('conf_mat.png', bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------
# DATASETS (tf.data, grayscale)
# ---------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Optional: performance tweaks
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ---------------------------------------------------------
# DATA AUGMENTATION (readable)
# ---------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomContrast(0.1),
])

# ---------------------------------------------------------
# STRONGER PURE CNN (VGG-ish, from scratch)
# ---------------------------------------------------------
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

x = data_augmentation(inputs)

# Block 1
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

# Block 2
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

# Block 3
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

# Block 4 (optional, but lightweight)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

# Replace Flatten with GAP
x = layers.GlobalAveragePooling2D()(x)

# Dense head
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=Adam(50e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#model.summary()

# ---------------------------------------------------------
# EARLY STOPPING
# ---------------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ---------------------------------------------------------
# EVALUATE ON TEST SET
# ---------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# ---------------------------------------------------------
# PREDICTIONS FOR CONFUSION MATRIX
# ---------------------------------------------------------
y_true_list = []
y_pred_list = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true_list.append(np.argmax(labels.numpy(), axis=1))
    y_pred_list.append(np.argmax(preds, axis=1))

y_true = np.concatenate(y_true_list)
y_pred = np.concatenate(y_pred_list)

conf_mat = get_conf_mat(y_true, y_pred, num_classes)
vis_conf_mat(conf_mat, class_names, test_acc)

print("Confusion matrix saved.")
