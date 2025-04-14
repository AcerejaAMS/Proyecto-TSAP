from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

class Propuesta_rgb:
    def __init__(self, images, label):
        # Reducitos tamaño para poder optimizar el uso de memoria
        images = tf.image.resize(images, [128, 128])
        images = tf.image.resize(images, [64, 64])
        images = tf.image.resize(images, [32, 32])
        
        # Normalizamos las imágenes al rango [0, 1]
        self.X_scaled = np.array(images / 255.0)
        self.y_onehot = np.array(to_categorical(label))

    def CrearModelo(self):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        # Creación del modelo
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.5)) 
        self.model.add(Dense(9, activation='softmax'))

        # Configuración del guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento del modelo
        self.history = self.model.fit(self.X_train, self.y_train, epochs=100,
                                    batch_size=16, validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo después del entrenamiento
        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
        else:
            print("El modelo no está cargado.")

class Propuesta_gray:
    def __init__(self, images, label):
        # Reducitos tamaño para poder optimizar el uso de memoria
        images = tf.image.resize(images, [128, 128])
        images = tf.image.resize(images, [64, 64])
        images = tf.image.resize(images, [32, 32])

        # Normalizamos las imágenes al rango [0, 1]
        self.X_scaled = np.array(images / 255.0)
        self.y_onehot = np.array(to_categorical(label))

    def CrearModelo(self):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        # Creación del modelo
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.5)) 
        self.model.add(Dense(9, activation='softmax'))

        # Configuración del guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento del modelo
        self.history = self.model.fit(self.X_train, self.y_train, epochs=100,
                                    batch_size=16, validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo después del entrenamiento
        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
        else:
            print("El modelo no está cargado.")