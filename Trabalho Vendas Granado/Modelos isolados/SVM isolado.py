import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Leitura dos dados
df = pd.read_csv('venda_por_ncm_e_estado.csv')
df.columns = df.columns.str.strip()  # Remove espaços dos nomes das colunas

# Remove colunas desnecessárias
df = df.drop(columns=['CFOP', 'NCM', 'ID utilização', 'Cod.Item'], errors='ignore')

# Remove linhas com dados faltantes nas colunas importantes
df = df.dropna(subset=['Valor', 'ICMS', 'Produto'])

# Filtra produtos com pelo menos 30 ocorrências
produto_counts = df['Produto'].value_counts()
produtos_validos = produto_counts[produto_counts >= 30].index

df = df[df['Produto'].isin(produtos_validos)]

# Codifica o alvo
le_produto = LabelEncoder()
y_encoded = le_produto.fit_transform(df['Produto'])

# Features e alvo
X = df[['Valor', 'ICMS']]

# Divisão treino/teste com balanceamento
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in splitter.split(X, y_encoded):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

# Pipeline com RobustScaler e SVC
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('svc', SVC(kernel='rbf', C=100, gamma=0.1, class_weight='balanced', random_state=42))
])

# Treinamento
pipeline.fit(X_train, y_train)

# Avaliação
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=le_produto.classes_, zero_division=0))

# Previsão com entrada do usuário
try:
    VALOR = float(input('Digite o valor (ex.: 3003.27): '))
    ICMS = float(input('Digite o valor do ICMS (ex.: 1522.23): '))
    entrada = pd.DataFrame([[VALOR, ICMS]], columns=['Valor', 'ICMS'])
    predicao = pipeline.predict(entrada)
    produto_previsto = le_produto.inverse_transform(predicao)[0]
    print(f"Previsão para o novo dado: {produto_previsto}")
except ValueError:
    print("Entrada inválida. Certifique-se de digitar números válidos.")
