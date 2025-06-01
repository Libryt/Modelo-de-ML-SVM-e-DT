import pandas as pd
import matplotlib.pyplot as plt # Lib para mostrar árvore de decisão
from sklearn import tree, svm # Lib para árvore e svm
from sklearn.preprocessing import LabelEncoder, RobustScaler #Lib para codificar
from sklearn.metrics import accuracy_score, classification_report #Lib para dar acurácia
from sklearn.model_selection import train_test_split # Lib para dividir
from sklearn.svm import SVC # Lib para o SVC


# Leitura dos dados
df = pd.read_csv('venda_por_ncm_e_estado.csv')
df.columns = df.columns.str.strip()  # Remove espaços dos nomes das colunas

# Remove colunas inúteis
df = df.drop(columns=['CFOP', 'NCM', 'ID utilização','Utilização', 'Cod.Item', 'UF'], axis = 1)
# Remove linhas com dados faltantes nas colunas importantes
df = df.dropna(subset=['Valor', 'ICMS', 'Produto'])

# Define as features e o alvo

X = df[['Valor', 'ICMS']]
y = df['Produto']

# Codifica o alvo (Produto)
le_produto = LabelEncoder()
y = le_produto.fit_transform(y)

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

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
                print("\nSeção SVM\nEscolha uma opção:")
                print("1 - Mostrar Desempenho\n2 - Fazer nova classificação\n3 - Voltar ao menu principal")
                try:
                    menuSVM = int(input("Digite uma opção: "))
                    if menuSVM == 1:
                        y_prediction = clf.predict(X_test_scaled)

                        print("Predição para o SVC: ", y_prediction)
                        acurácia = accuracy_score(y_test, y_prediction)
                        print(f"Acurácia: {acurácia:.2f}") 
                        
                        print("\nRelatório de Classificação Detalhado:")
                        print(classification_report(y_test, y_prediction, target_names=le_produto.classes_, zero_division=0))

                    elif menuSVM == 2:
                        # Previsão com entrada do usuário (adaptado para não usar pipeline)
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
