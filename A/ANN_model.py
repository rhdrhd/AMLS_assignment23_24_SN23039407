import numpy as np
import random
import tensorflow as tf
import wandb
from keras import layers,models
from utils import *
import os
from tensorflow.keras.optimizers import Adam, SGD, Adamax
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score


#fix random seeds to increase reproducibility 42
random.seed(23)
np.random.seed(23)
tf.random.set_seed(23)

x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_dataset()

x_train = x_train.reshape((-1, 28, 28))
x_val = x_val.reshape((-1, 28, 28))
x_test = x_test.reshape((-1, 28, 28))

#x_train = np.array([random_rotate(image) for image in x_train])
#x_train = np.array([add_noise(image, noise_type="gaussian") for image in x_train])

sweep_config = {
    'method': 'bayes',  # Choose 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'   
    },
    'parameters': {
        'num_conv_layers': {
            'values': [1, 2, 3]  # Number of Convolutional Layers
        },
        'max_pooling': {
            'values': [True, False]  # Use of MaxPooling
        },
        'num_neurons': {
            'values': [16, 32, 64, 128]  # Neurons in Dense Layer
        },
        'dropout_rate': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5]  # Dropout rate
        }
    }
}



def sweep_train():
    # Initialize a new wandb run
    wandb.init()
    
    # Access hyperparameters
    config = wandb.config

    model = models.Sequential()
    model.add(layers.Conv2D(8, (2, 2), activation='relu', input_shape=(28, 28, 1)))

    for _ in range(config.num_conv_layers - 1):
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        if config.max_pooling:
            model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(config.dropout_rate))

    model.add(layers.Flatten())
    model.add(layers.Dense(config.num_neurons, activation='relu'))
    model.add(layers.Dropout(config.dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), callbacks=[wandb.keras.WandbCallback()])

    # Save the model with wandb
    model.save(os.path.join(wandb.run.dir, "model.h5"))

def perform_sweep_wandb():
    wandb.login()
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, sweep_train)

def train_selected_model():

    model = models.Sequential()

    # Add convolutional layers
    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Adjust input_shape based on your data
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(16, (2, 2), activation='relu'))
    model.add(layers.Dropout(0.25))

    # Flatten the output from 2D to 1D
    model.add(layers.Flatten())

    # Add dense layers (fully connected layers)
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    # Fit the model
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_val, y_val), callbacks = [checkpoint])
    best_model = load_model('best_model.h5')
    # Evaluate the model
    test_score = best_model.evaluate(x_test, y_test)[1]
    train_score = best_model.evaluate(x_train, y_train)[1]
    print(f"Score on test: {test_score}")
    print(f"Score on train: {train_score}")


def load_and_test_model(path):
    best_model = load_model(path)

    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)

    y_pred_probs = best_model.predict(x_test)
    auc = roc_auc_score(y_test, y_pred_probs)

    print(f"Accuracy on test: {test_accuracy}")
    print(f"Test AUC: {auc}")


load_and_test_model("best_model_88_94.h5")