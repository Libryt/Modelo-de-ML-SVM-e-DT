import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Lib para mostrar árvore de decisão
from sklearn import tree, svm # Lib para árvore e svm
from sklearn.preprocessing import LabelEncoder, RobustScaler #Lib para codificar
from sklearn.metrics import accuracy_score, classification_report #Lib para dar acurácia
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit # Lib para dividir
from sklearn.pipeline import Pipeline #Lib para o pipeline
from sklearn.svm import SVC # Lib para o SVC

#TODO Tirar pipeline do SVM

# Leitura dos dados
df = pd.read_csv('venda_por_ncm_e_estado.csv')
df.columns = df.columns.str.strip()  # Remove espaços dos nomes das colunas

# Remove colunas inúteis
df = df.drop(columns=['CFOP', 'NCM', 'ID utilização','Utilização', 'Cod.Item', 'UF'])
#df = df.drop('Produto', axis=1)
# Remove linhas com dados faltantes nas colunas importantes
df = df.dropna(subset=['Valor', 'ICMS', 'Produto'])

# Define as features e o alvo

X = df[['Valor', 'ICMS']]
#TESTAR ESSE TAMBEM X = df.drop('Produto', axis=1)
y = df['Produto']

# Codifica o alvo (Produto)
le_produto = LabelEncoder()
y = le_produto.fit_transform(y)

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Loop principal do menu
while True:
    print("\nMenu principal\nEscolha uma opção: \n1 - Árvore de Decisão\n2 - SVM\n3 - Encerrar programa")
    try:
        menu = int(input("Digite uma opção: "))
        # Aqui vai a árvore de decisão
        if menu == 1:
            clf = tree.DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            while True:
                print("\nSeção Árvore de Decisão\nEscolha uma opção:")
                print("1 - Mostrar Desempenho\n2 - Mostrar Árvore\n3 - Fazer nova classificação\n4 - Voltar ao menu principal")
                try:
                    menuDecisao = int(input("Digite uma opção: "))
                    if menuDecisao == 1:
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        print(f"Precisão do modelo: {acc:.2f}")

                    elif menuDecisao == 2:
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

                    elif menuDecisao == 3:
                        try:
                            VALOR = float(input("Digite o valor (ex.: 3003.27): "))
                            ICMS = float(input("Digite o valor do ICMS (ex.: 1522.23): "))
                            entrada = pd.DataFrame([[VALOR, ICMS]], columns=['Valor', 'ICMS'])
                            previsao = clf.predict(entrada)
                            produto_previsto = le_produto.inverse_transform([previsao[0]])[0]
                            print("Previsão do produto:", produto_previsto)
                        except ValueError:
                            print("Entrada inválida. Digite números válidos.")

                    elif menuDecisao == 4:
                        print("Voltando ao menu principal.")
                        break

                    else:
                        print("Opção inválida. Tente novamente.")
                except ValueError:
                    print("Entrada inválida. Digite apenas números inteiros.")
        # AQUI VAI O SVM
        elif menu == 2:
            model = svm.SVC()
            model.fit(X_train, y_train)
            while True:
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
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
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
                print("\nSeção SVM\nEscolha uma opção:")
                print("1 - Mostrar Desempenho\n2 - Fazer nova classificação\n3 - Voltar ao menu principal")
                try:
                    menuSVM = int(input("Digite uma opção: "))
                    if menuSVM == 1:
                        y_pred = pipeline.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        print(f"Acurácia do modelo: {accuracy:.2f}")
                        print("\nRelatório de Classificação:")
                        print(classification_report(y_test, y_pred, target_names=le_produto.classes_, zero_division=0))

                    elif menuSVM == 2:
                        try:
                            VALOR = float(input('Digite o valor (ex.: 3003.27): '))
                            ICMS = float(input('Digite o valor do ICMS (ex.: 1522.23): '))
                            entrada = pd.DataFrame([[VALOR, ICMS]], columns=['Valor', 'ICMS'])
                            predicao = pipeline.predict(entrada)
                            produto_previsto = le_produto.inverse_transform(predicao)[0]
                            print(f"\nPrevisão para o novo dado: {produto_previsto}")
                        except ValueError:
                            print("Entrada inválida. Certifique-se de digitar números válidos.")

                    elif menuSVM == 3:
                        print("Voltando ao menu principal.")
                        break

                    else:
                        print("Opção inválida. Tente novamente.")
                except ValueError:
                    print("Entrada inválida. Digite apenas números inteiros.")

        elif menu == 3:
            print("Programa encerrado com sucesso.")
            break

        else:
            print("Opção inválida. Tente novamente.")
    except ValueError:
        print("Entrada inválida. Por favor, digite um número inteiro.")
