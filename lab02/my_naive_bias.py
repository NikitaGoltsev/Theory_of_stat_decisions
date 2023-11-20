import numpy as np


class Naive_Bayes:
    def __init__(self):
        # Словарь для хранения априорных вероятностей классов
        self.class_probabilities = {}
        # Словарь для хранения средних значений и дисперсий признаков
        self.mean_variances = {}
        # Список уникальных классов
        self.classes = []

    def fit(self, X, y):
        # Получение уникальных классов в целевой переменной
        self.classes = np.unique(y)
        # Общее количество обучающих примеров
        total_samples = len(y)
        for class_label in self.classes:
            # Выбор обучающих примеров для данного класса
            class_samples = X[y == class_label]
            # Вычисление априорной вероятности класса
            self.class_probabilities[class_label] = len(class_samples) / total_samples
            # Список для хранения средних значений и дисперсий признаков
            self.mean_variances[class_label] = []
            for feature_index in range(X.shape[1]):
                # Выбор значений признака для данного класса
                feature_values = class_samples[:, feature_index]
                # Вычисление среднего и дисперсии значений признака
                mean = np.mean(feature_values)
                variance = np.var(feature_values)
                self.mean_variances[class_label].append((mean, variance))

    def calculate_likelihood(self, x, mean, variance):
        # Вычисление плотности вероятности для нормального распределения
        epsilon = 1e-16
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance + epsilon))
        likehood = (exponent / (np.sqrt(2 * np.pi * variance) + epsilon))
        return likehood

    def ___calculate_likelihood(self, x, mean, variance):
        epsilon = 1e-16
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance + epsilon))
        likehood = (exponent / (np.sqrt(2 * np.pi * variance) + epsilon))
        likehood = np.clip(likehood, epsilon, None)
        return likehood

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = {}
            for class_label in self.classes:
                # Получение априорной вероятности класса
                class_probability = self.class_probabilities[class_label]
                # Вычисление плотностей вероятности для признаков и их произведение
                likelihoods = np.array([self.calculate_likelihood(sample[i], mean, variance)
                                        for i, (mean, variance) in enumerate(self.mean_variances[class_label])])
                # Вычисление апостериорной вероятности для класса
                posterior = class_probability * np.prod(likelihoods)
                class_scores[class_label] = posterior
            # Выбор класса с наибольшей апостериорной вероятностью
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions
