import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Fase 2 - Carregamento

def carregar_dados(caminho):
    try:
        dados = pd.read_csv(caminho)
        return dados
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

# Fase 3 - Preparacao

def preparar_dados(df):
    df = df.drop_duplicates()

    # Converter coluna alvo para binária
    df['Class'] = df['Class'].apply(lambda x: 1 if x == 'L' else 0)

    # Codificar variáveis categóricas
    for coluna in df.select_dtypes(include='object').columns:
        df[coluna] = LabelEncoder().fit_transform(df[coluna])

    return df

# Fase 3 - Visualização

def visualizar_alvo(df):
    df['Class'].value_counts().plot(kind='bar', title="Distribuição da Classe Alvo (Risco de Evasão)")
    plt.xticks([0, 1], ['Sem risco', 'Risco'])
    plt.show()

# Fase 4 - Modelagem

def dividir_dados(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

def treinar_modelos(x_train, y_train):
    modelos = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        LogisticRegression(max_iter=1000),
        GaussianNB()
    ]
    treinados = [modelo.fit(x_train, y_train) for modelo in modelos]
    return treinados

# Fase 5 - Avaliação

def avaliar_modelos(modelos, x_test, y_test):
    acuracias = [accuracy_score(y_test, modelo.predict(x_test)) for modelo in modelos]
    melhor_indice = np.argmax(acuracias)
    print("Acurácias:", acuracias)
    print(f"Melhor modelo: {type(modelos[melhor_indice]).__name__} com {acuracias[melhor_indice]*100:.2f}%")
    return modelos[melhor_indice]

# Fase 6 - Implantação

def salvar_modelo(nome_arquivo, modelo):
    with open(nome_arquivo, 'wb') as file:
        pickle.dump(modelo, file)

def carregar_modelo(nome_arquivo):
    with open(nome_arquivo, 'rb') as file:
        return pickle.load(file)

def validar_modelo(modelo, x_val):
    pred = modelo.predict(x_val)
    print("Previsões do modelo:", pred)

# Execução principal
if __name__ == "__main__":
    dados = carregar_dados("xAPI-Edu-Data.csv")
    if dados is not None:
        dados = preparar_dados(dados)
        visualizar_alvo(dados)
        x_train, x_test, y_train, y_test = dividir_dados(dados)
        modelos = treinar_modelos(x_train, y_train)
        melhor_modelo = avaliar_modelos(modelos, x_test, y_test)
        salvar_modelo("modelo_evasao.pickle", melhor_modelo)
        modelo_carregado = carregar_modelo("modelo_evasao.pickle")
        validar_modelo(modelo_carregado, x_test)
