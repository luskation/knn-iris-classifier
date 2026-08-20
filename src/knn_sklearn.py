"""
KNN utilizando a implementação pronta do Scikit-learn, para comparação
com a versão hardcore (knn_scratch.py). Usa exatamente o mesmo split
treino/teste para garantir uma comparação justa.
"""

import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


def carregar_dados(test_size=0.3, random_state=42):
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, iris.target_names


def executar():
    X_train, X_test, y_train, y_test, nomes_classes = carregar_dados()

    resultados = {}
    inicio = time.time()

    for k in [1, 3, 5, 7]:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        matriz = confusion_matrix(y_test, y_pred)
        acuracia = accuracy_score(y_test, y_pred)
        precisao = precision_score(y_test, y_pred, average="macro")
        revocacao = recall_score(y_test, y_pred, average="macro")

        resultados[k] = {
            "matriz": matriz,
            "acuracia": acuracia,
            "precisao_media": precisao,
            "revocacao_media": revocacao,
        }

        print(f"\n===== k = {k} (SKLEARN) =====")
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Precisão média: {precisao:.4f}")
        print(f"Revocação média: {revocacao:.4f}")
        print("Matriz de confusão:")
        print(matriz)

    tempo_total = time.time() - inicio
    print(f"\nTempo total de execução (4 valores de k): {tempo_total:.4f}s")

    return resultados, tempo_total


if __name__ == "__main__":
    executar()
