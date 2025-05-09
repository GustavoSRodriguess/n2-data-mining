# -*- coding: utf-8 -*-
"""questao3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FezjbBGCbiySHvOMMb2Dg5B1USrXozod
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

#alunos: gustavo schneider rodrigues, felipe beppler huller

df = pd.read_csv('StudentsPerformance.csv')

df['pass_math'] = df['math score'].apply(lambda x: 1 if x >= 70 else 0)

X = df[['reading score', 'writing score', 'gender']]
y = df['pass_math']

label_encoder = LabelEncoder()
X['gender'] = label_encoder.fit_transform(X['gender'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
log_pred_prob = log_model.predict_proba(X_test_scaled)[:, 1]

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
lin_pred_raw = lin_model.predict(X_test_scaled)
lin_pred = np.where(lin_pred_raw >= 0.5, 1, 0)

print("=== Regressão Logística ===")
print(f"Acurácia: {accuracy_score(y_test, log_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, log_pred_prob):.4f}")
print("Relatório de Classificação:")
print(classification_report(y_test, log_pred))

print("\n=== Regressão Linear ===")
print(f"MSE: {mean_squared_error(y_test, lin_pred_raw):.4f}")
print(f"Acurácia: {accuracy_score(y_test, lin_pred):.4f}")
print("Relatório de Classificação:")
print(classification_report(y_test, lin_pred))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Real', alpha=0.5)
plt.scatter(range(len(lin_pred_raw)), lin_pred_raw, color='red', label='Predição Linear', alpha=0.5)
plt.axhline(y=0.5, color='green', linestyle='--', label='Limite de Decisão')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='black', linestyle='-', alpha=0.3)
plt.title('Predições da Regressão Linear')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Real', alpha=0.5)
plt.scatter(range(len(log_pred_prob)), log_pred_prob, color='purple', label='Probabilidades Logísticas', alpha=0.5)
plt.axhline(y=0.5, color='green', linestyle='--', label='Limite de Decisão')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='black', linestyle='-', alpha=0.3)
plt.title('Predições da Regressão Logística')
plt.legend()

plt.tight_layout()
plt.show()

#alunos: gustavo schneider rodrigues, felipe beppler huller
if X.shape[1] >= 2:
    plt.figure(figsize=(12, 5))

    feature1, feature2 = 0, 1

    x_min, x_max = X_test_scaled[:, feature1].min() - 1, X_test_scaled[:, feature1].max() + 1
    y_min, y_max = X_test_scaled[:, feature2].min() - 1, X_test_scaled[:, feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    mesh_points = np.zeros((xx.ravel().shape[0], X_test_scaled.shape[1]))
    mesh_points[:, feature1] = xx.ravel()
    mesh_points[:, feature2] = yy.ravel()

    plt.subplot(1, 2, 1)
    Z_log = log_model.predict(mesh_points).reshape(xx.shape)
    plt.contourf(xx, yy, Z_log, alpha=0.4, cmap='RdBu')
    plt.scatter(X_test_scaled[:, feature1], X_test_scaled[:, feature2], c=y_test, cmap='RdBu', edgecolors='k')
    plt.title('Fronteira de Decisão - Regressão Logística')
    plt.xlabel(f'Feature {feature1} (Escalada)')
    plt.ylabel(f'Feature {feature2} (Escalada)')

    plt.subplot(1, 2, 2)
    Z_lin = (lin_model.predict(mesh_points) >= 0.5).reshape(xx.shape)
    plt.contourf(xx, yy, Z_lin, alpha=0.4, cmap='RdBu')
    plt.scatter(X_test_scaled[:, feature1], X_test_scaled[:, feature2], c=y_test, cmap='RdBu', edgecolors='k')
    plt.title('Fronteira de Decisão - Regressão Linear')
    plt.xlabel(f'Feature {feature1} (Escalada)')
    plt.ylabel(f'Feature {feature2} (Escalada)')

    plt.tight_layout()
    plt.show()

"""a. Problema:

Prever se um aluno passou ou não em Matemática com base nas notas de Leitura, Redação e no gênero.
b. Dados utilizados para modelagem do problema (dataset):

Utilizou-se o dataset "StudentsPerformance.csv", contendo as variáveis:

    math score (nota de Matemática)

    reading score (nota de Leitura)

    writing score (nota de Redação)

    gender (gênero do aluno)

A variável target pass_math foi criada a partir da nota de Matemática (1 para notas >= 70, 0 para notas < 70).
c. Treinamento do modelo:

Foram treinados dois modelos:

    Regressão Logística, para classificação binária.

    Regressão Linear, convertendo as previsões contínuas em binárias.

Os modelos foram treinados com 70% dos dados e testados com 30%.
d. Resultado do modelo:

    Regressão Logística: 85% de acurácia, AUC = 0.9482.

    Regressão Linear: 84.33% de acurácia, desempenho inferior em relação à regressão logística.

Em resumo, a Regressão Logística foi mais eficaz para este problema de classificação binária.
"""