from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

class Propuesta_128:
    def __init__(self, images, labels):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        # del images, labels  # Liberar memoria

    def CrearModelo(self): # Agragra opcion de opti, activacion, dropout, AVreagePooling,
        # Creación del modelo
        self.model = Sequential()

        # Capa Convolucional 1
        self.model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu', kernel_regularizer=l2(0.001), input_shape=(128, 128, 3)))
        self.model.add(BatchNormalization())
        # self.model.add(Leakyelu(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Capa Convolucional 2
        self.model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', kernel_regularizer=l2(0.001)))
        self.model.add(BatchNormalization())
        # self.model.add(Leakyelu(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Capa Convolucional 3
        self.model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu', kernel_regularizer=l2(0.001)))
        self.model.add(BatchNormalization())
        # self.model.add(Leakyelu(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Aplanamiento
        self.model.add(GlobalMaxPooling2D())
        

        # Capas Densa 1
        self.model.add(Dense(128, kernel_regularizer=l2(0.001)))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(0.4))

        # Capas Densa 2
        self.model.add(Dense(64, kernel_regularizer=l2(0.001)))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(0.4))

        # # Capas Densa 3
        # self.model.add(Dense(32, activation = 'elu', kernel_regularizer=l2(0.001)))
        # # self.model.add(Leakyelu(alpha=0.1))
        # self.model.add(Dropout(0.4)) 

        # Capas Salida
        self.model.add(Dense(9, activation='softmax'))

        # Configuración del guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0005) # Recomendado para un dataset grande y ruidoso
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento del modelo
        self.history = self.model.fit(self.X_train, self.y_train, epochs=100,
                                    batch_size=32, validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo después del entrenamiento
        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
            return self.accuracy
        else:
            print("El modelo no está cargado.")

####################################################################################################################################################################
class Propuesta_224:
    def __init__(self, images, labels):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        del images, labels  # Liberar memoria

    def CrearModelo(self):
        # Creación del modelo
        self.model = Sequential()

        # Capa Convolucional 1
        self.model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu', kernel_regularizer=l2(0.001), input_shape=(224, 224, 3)))
        self.model.add(BatchNormalization())
        # self.model.add(Leakyelu(alpha=0.1))
        self.model.add(AveragePooling2D(pool_size=(2, 2)))

        # Capa Convolucional 2
        self.model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', kernel_regularizer=l2(0.001)))
        self.model.add(BatchNormalization())
        # self.model.add(Leakyelu(alpha=0.1))
        self.model.add(AveragePooling2D(pool_size=(2, 2)))

        # Capa Convolucional 3
        self.model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu', kernel_regularizer=l2(0.001)))
        self.model.add(BatchNormalization())
        # self.model.add(Leakyelu(alpha=0.1))
        self.model.add(AveragePooling2D(pool_size=(2, 2)))

        # Aplanamiento
        self.model.add(GlobalAveragePooling2D())

        # Capas Densa 1
        self.model.add(Dense(128, kernel_regularizer=l2(0.001)))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(0.4))

        # Capas Densa 2
        self.model.add(Dense(64, kernel_regularizer=l2(0.001)))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(0.4))

        # # Capas Salida
        # self.model.add(Dense(32, kernel_regularizer=l2(0.001)))
        # # self.model.add(Leakyelu(alpha=0.1))
        # self.model.add(Dropout(0.4)) 

        self.model.add(Dense(9, activation='softmax'))

        # Configuración del guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0005) # Recomendado para un dataset grande y ruidoso
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento del modelo
        self.history = self.model.fit(self.X_train, self.y_train, epochs=100,
                                    batch_size=32, validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo después del entrenamiento
        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
        else:
            print("El modelo no está cargado.")