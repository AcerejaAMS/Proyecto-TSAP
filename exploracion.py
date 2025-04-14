import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class Conteo_por_etiquetas:
    def __init__(self, labels, label_descriptions):
        """
        labels: Etiquetas numericas
        label_descriptions: Diccionario con las descripciones por etiqueta
        """
        self.labels = pd.Series(labels)
        self.labels_originales = pd.Series(labels)
        self.label_descriptions = label_descriptions

        # Reemplazar las etiquetas numéricas por las descripciones
        self.labels_mapeadas = self.labels.map(label_descriptions)

        # Contar la frecuencia de cada etiqueta
        self.label_counts = self.labels_mapeadas.value_counts().reset_index()
        self.label_counts.columns = ['label', 'count']

    def mostrar(self):
        # Graficar el histograma usando Seaborn
        plt.figure(figsize=(12, 10))
        sns.barplot(x='label', y='count', data=self.label_counts, palette='colorblind')

        # Mejorar el diseño
        plt.title('Histograma por DeepWeeds', fontsize=20, weight='bold')
        plt.xlabel('DeepWeeds', fontsize=14)
        plt.ylabel('Conteo', fontsize=14)
        plt.xticks(rotation=50, ha='right', fontsize=10)  # Rotar las etiquetas para mejor legibilidad

        # Mostrar la gráfica
        plt.tight_layout()
        plt.show()

class Visualizacion_por_canal:
    def __init__(self, rgb, gray, labels):
        self.labels = pd.Series(labels)
        self.images_rgb = rgb
        self.images_gray = gray

        self.red_means = []
        self.green_means = []
        self.blue_means = []
        self.gray_means = []
        self.labels_m = []

        i = 0

        for img_rgb, img_gray in zip(self.images_rgb, self.images_gray):
            self.red_mean = img_rgb[:, :, 0].mean()
            self.green_mean = img_rgb[:, :, 1].mean()
            self.blue_mean = img_rgb[:, :, 2].mean()
            self.gray_mean = img_gray[:, :, 0].mean()

            self.red_means.append(self.red_mean)
            self.green_means.append(self.green_mean)
            self.blue_means.append(self.blue_mean)
            self.gray_means.append(self.gray_mean)
            self.labels_m.append(self.labels[i].item())
            
            i += 1

        self.df = pd.DataFrame({
                'Red_mean': self.red_means,
                'Green_mean': self.green_means,
                'Blue_mean': self.blue_means,
                'Gray_mean': self.gray_means,
                'label': self.labels_m
            })
    
    def informacion_promedio_color(self):
        sns.pairplot(self.df, hue='label', diag_kind='kde')
        plt.show()
    
    def distribucion_canal_etiqueta(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 20))

        # 1. Gráfico para el canal Rojo (Red_mean)
        sns.boxplot(x='label', y='Red_mean', data=self.df, palette='Reds', ax=axs[0])
        axs[0].set_title('Distribución de Red_mean por Etiqueta', fontsize=14)
        axs[0].set_xlabel('Etiqueta', fontsize=12)
        axs[0].set_ylabel('Valor medio de Red_mean', fontsize=12)

        # 2. Gráfico para el canal Verde (Green_mean)
        sns.boxplot(x='label', y='Green_mean', data=self.df, palette='Greens', ax=axs[1])
        axs[1].set_title('Distribución de Green_mean por Etiqueta', fontsize=14)
        axs[1].set_xlabel('Etiqueta', fontsize=12)
        axs[1].set_ylabel('Valor medio de Green_mean', fontsize=12)

        # 3. Gráfico para el canal Azul (Blue_mean)
        sns.boxplot(x='label', y='Blue_mean', data=self.df, palette='Blues', ax=axs[2])
        axs[2].set_title('Distribución de Blue_mean por Etiqueta', fontsize=14)
        axs[2].set_xlabel('Etiqueta', fontsize=12)
        axs[2].set_ylabel('Valor medio de Blue_mean', fontsize=12)

        # 4. Gráfico para el canal Gris (Gray_mean)
        sns.boxplot(x='label', y='Gray_mean', data=self.df, palette='Greys', ax=axs[3])
        axs[3].set_title('Distribución de Gray_mean por Etiqueta', fontsize=14)
        axs[3].set_xlabel('Etiqueta', fontsize=12)
        axs[3].set_ylabel('Valor medio de Gray_mean', fontsize=12)

        plt.tight_layout()
        plt.show()
