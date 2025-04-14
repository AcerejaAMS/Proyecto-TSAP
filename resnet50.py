import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

class ResNet50_rgb:
    def __init__(self, images, labels):
        # Reducitos tamaño para poder optimizar el uso de memoria
        images = tf.image.resize(images, [128, 128])
        images = tf.image.resize(images, [64, 64])
        images = tf.image.resize(images, [32, 32])

        # Normalizamos las imágenes al rango [0, 1]
        self.X_scaled = np.array(images / 255.0)
        self.y_onehot = np.array(to_categorical(labels))

    def CrearModelo(self):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        # Cargar la base de ResNet50 sin las capas superiores
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(32, 32, 3)))

        # Congelamos las capas de ResNet50 para que no se entrenen
        for layer in base_model.layers:
            layer.trainable = False

        # Añadimos nuestras propias capas densas
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(9, activation='softmax')(x)

        # Creamos el modelo completo
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento
        self.history = self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=16,
                                    validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo
        self.model = tf.keras.models.load_model('model.keras')

        return self.model

    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy * 100:.2f}%')
        else:
            print("El modelo no está cargado.")

class ResNet50_gray:
    def __init__(self, images, labels):
        # Reducitos tamaño para poder optimizar el uso de memoria
        images = tf.image.resize(images, [128, 128])
        images = tf.image.resize(images, [64, 64])
        images = tf.image.resize(images, [32, 32])
        
        # Normalizamos las imágenes al rango [0, 1]
        self.X_scaled = np.array(images / 255.0)
        self.X_scaled = np.repeat(self.X_scaled, 3, axis=-1)
        self.y_onehot = np.array(to_categorical(labels))

    def CrearModelo(self):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        # Cargar la base de ResNet50 sin las capas superiores
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(32, 32, 3)))

        # Congelamos las capas de ResNet50 para que no se entrenen
        for layer in base_model.layers:
            layer.trainable = False

        # Añadimos nuestras propias capas densas
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(9, activation='softmax')(x)

        # Creamos el modelo completo
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento
        self.history = self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=16,
                                    validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo
        self.model = tf.keras.models.load_model('model.keras')

        return self.model

    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy * 100:.2f}%')
        else:
            print("El modelo no está cargado.")