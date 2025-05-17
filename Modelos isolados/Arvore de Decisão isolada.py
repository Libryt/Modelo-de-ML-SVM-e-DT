import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Leitura dos dados
df = pd.read_csv('venda_por_ncm_e_estado.csv')
df.columns = df.columns.str.strip()  # Remove espaços dos nomes das colunas

# Remove colunas inúteis
df = df.drop(columns=['NCM', 'ID utilização', 'Cod.Item', 'UF'])

# Remove linhas com dados faltantes nas colunas importantes
df = df.dropna(subset=['Valor', 'ICMS', 'Produto'])

# Define as features e o alvo
X = df[['Valor', 'ICMS', 'CFOP']]
y = df['Produto']

# Codifica o alvo (Produto)
le_produto = LabelEncoder()
y = le_produto.fit_transform(y)

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treina o modelo
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Avalia o modelo e mostra acurácia
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisão do modelo: {accuracy:.2f}%")

# Entrada do usuário para previsão
VALOR = float(input('Digite o valor (ex.: 3003.27): '))
ICMS = float(input('Digite o valor do ICMS (ex.: 1522.23): '))
CFOP = int(input("Digite o núemero CFOP(5102 ou 5405)"))

# Cria Dataframe para prever
df_para_classificar = pd.DataFrame([[VALOR, ICMS]], columns=['Valor', 'ICMS'])

# Aqui eu mostro a previsão do produto 
previsao = clf.predict(df_para_classificar)
produto_previsto = le_produto.inverse_transform([previsao[0]])[0]
print("Previsão do produto:", produto_previsto)

# Aqui eu mostro a árvore de decisão

plt.figure(figsize=(20, 10))
tree.plot_tree(
    clf,
    feature_names=['Valor', 'ICMS'],
    class_names=le_produto.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árvore de Decisão - Classificação de Produto")
plt.show()
