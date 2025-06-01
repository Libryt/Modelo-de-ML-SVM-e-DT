import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
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

# Codificação do alvo e definição das features
le_produto = LabelEncoder()
df['Produto_Encoded'] = le_produto.fit_transform(df['Produto']) # Criar coluna para y

# Definição dos dados de entrada X e saída y
X = df[['Valor', 'ICMS']]
y = df['Produto_Encoded'] # Usar a coluna codificada

# Divisão dos dados em treinamento e teste (semelhante ao primeiro script)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#Escalonamento dos dados
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aprendizado
clf = SVC(kernel='rbf', C=100, gamma=0.1, class_weight='balanced', random_state=1)
clf.fit(X_train_scaled, y_train)

# Mostrar desempenho

y_prediction = clf.predict(X_test_scaled)

print("Predição para o SVC: ", y_prediction)
acurácia = accuracy_score(y_test, y_prediction)
print(f"Acurácia: {acurácia:.2f}") 

print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_prediction, target_names=le_produto.classes_, zero_division=0))

# Previsão com entrada do usuário
try:
    VALOR = float(input('Digite o valor (ex.: 3003.27): '))
    ICMS = float(input('Digite o valor do ICMS (ex.: 1522.23): '))
    entrada = pd.DataFrame([[VALOR, ICMS]], columns=['Valor', 'ICMS'])
    
    # Escalonar a entrada do usuário com o mesmo scaler usado no treino
    entrada_scaled = scaler.transform(entrada)
    
    # Realizar a predição com o modelo treinado
    predicao_codificada = clf.predict(entrada_scaled)
    
    # Decodificar a predição para o nome original do produto
    produto_previsto = le_produto.inverse_transform(predicao_codificada)[0]
    
    print(f"Previsão para o novo dado: {produto_previsto}")

except ValueError:
    print("Entrada inválida. Certifique-se de digitar números válidos.")
except Exception as e:
    print(f"Ocorreu um erro durante a previsão: {e}")