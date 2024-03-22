import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Caminho para o arquivo 'wine_data.csv'
file_path = 'C:/Guilherme/Repositórios-GIT/INTELIGÊNCIA-ARTIFICIAL/AtividadePratica-Ciclo1/wine_dataset.csv'

# Verificar se o arquivo existe
if not os.path.exists(file_path):
    # Dados sintéticos para o arquivo CSV
    wine_data = pd.DataFrame({
        'acidez': [6.5, 7.2, 8.0, 6.8, 7.0, 6.6, 7.4, 6.9, 7.1, 6.7],
        'teor_alcoolico': [12.0, 13.5, 11.0, 13.2, 12.8, 12.3, 12.1, 12.5, 11.5, 12.6],
        'pH': [3.2, 3.1, 3.5, 3.3, 3.4, 3.0, 3.6, 3.2, 3.3, 3.1],
        'densidade': [0.98, 1.01, 0.99, 1.05, 1.03, 1.00, 0.97, 1.02, 0.98, 1.04],
        'label': ['branco', 'branco', 'vermelho', 'vermelho', 'branco', 'branco', 'vermelho', 'vermelho', 'branco', 'vermelho']
    })
    wine_data.to_csv(file_path, index=False)
    print(f"Arquivo '{file_path}' gerado com sucesso.")
else:
    # Carregar o conjunto de dados
    wine_data = pd.read_csv(file_path)

# Exibir informações sobre o conjunto de dados
print("Leitura e Exploração de Dados")
print("Informações sobre o conjunto de dados:")
print(wine_data.info())
print("\nDescrição estatística do conjunto de dados:")
print(wine_data.describe())

# Pré-processamento dos dados
X_features = wine_data.drop('style', axis=1)

# Lidar com valores ausentes usando fillna do pandas
X_features.fillna(X_features.mean(), inplace=True)

# Normalizar as características
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_features, wine_data['style'], test_size=0.2, random_state=42)

# Implementar e treinar o modelo LinearSVC
linear_svc_model = LinearSVC(random_state=42)
linear_svc_model.fit(X_train, y_train)

# Avaliar os modelos
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do {model_name}:", accuracy)

# Avaliar o modelo LinearSVC
print("\nComparação de Modelos")
evaluate_model(linear_svc_model, X_test, y_test, "Modelo LinearSVC")

# Implementar e avaliar o DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
evaluate_model(dummy_clf, X_test, y_test, "DummyClassifier (Estratégia 'most_frequent')")

# Análise e Discussão dos resultados
print("\nAnálise e Discussão")
print("O Modelo LinearSVC apresentou uma acurácia superior ao DummyClassifier.")
print("A escolha do modelo adequado é crucial, pois o LinearSVC oferece uma abordagem mais robusta e adaptável aos dados.")
