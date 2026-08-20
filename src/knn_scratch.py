"""
KNN implementado do zero (hardcore) para a base de dados Iris.
Nenhuma biblioteca de classificação pronta é utilizada — apenas NumPy
para operações vetoriais/numéricas e Pandas/Sklearn somente para
carregar os dados e fazer o split treino/teste.
"""

import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Carregamento e preparação dos dados
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Distância euclidiana
# ---------------------------------------------------------------------------
def distancia_euclidiana(p, q):
    return np.sqrt(np.sum((p - q) ** 2))


# ---------------------------------------------------------------------------
# 3 e 4. Busca dos k vizinhos mais próximos + votação majoritária (com desempate)
# ---------------------------------------------------------------------------
def prever_um_ponto(X_train, y_train, ponto, k):
    distancias = np.array([distancia_euclidiana(ponto, x) for x in X_train])
    indices_ordenados = np.argsort(distancias)
    k_indices = indices_ordenados[:k]
    k_classes = y_train[k_indices]
    k_distancias = distancias[k_indices]

    classes, contagens = np.unique(k_classes, return_counts=True)
    max_contagem = contagens.max()
    candidatas = classes[contagens == max_contagem]

    if len(candidatas) == 1:
        return candidatas[0]

    # Desempate: escolhe, entre as classes empatadas, a do vizinho mais próximo
    for idx in np.argsort(k_distancias):
        if k_classes[idx] in candidatas:
            return k_classes[idx]


# ---------------------------------------------------------------------------
# 5. Predição para um conjunto de pontos de teste
# ---------------------------------------------------------------------------
def prever(X_train, y_train, X_test, k):
    return np.array([prever_um_ponto(X_train, y_train, p, k) for p in X_test])


# ---------------------------------------------------------------------------
# 7. Matriz de confusão e métricas (implementadas na mão)
# ---------------------------------------------------------------------------
def matriz_confusao(y_true, y_pred, n_classes):
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    for real, previsto in zip(y_true, y_pred):
        matriz[real, previsto] += 1
    return matriz


def metricas(matriz):
    n_classes = matriz.shape[0]
    precisao, revocacao = [], []

    for c in range(n_classes):
        vp = matriz[c, c]
        fp = matriz[:, c].sum() - vp
        fn = matriz[c, :].sum() - vp

        precisao.append(vp / (vp + fp) if (vp + fp) > 0 else 0.0)
        revocacao.append(vp / (vp + fn) if (vp + fn) > 0 else 0.0)

    acuracia = np.trace(matriz) / matriz.sum()
    return {
        "acuracia": acuracia,
        "precisao_por_classe": precisao,
        "revocacao_por_classe": revocacao,
        "precisao_media": float(np.mean(precisao)),
        "revocacao_media": float(np.mean(revocacao)),
    }


# ---------------------------------------------------------------------------
# 6. Execução para k = {1, 3, 5, 7}
# ---------------------------------------------------------------------------
def executar():
    X_train, X_test, y_train, y_test, nomes_classes = carregar_dados()
    n_classes = len(nomes_classes)

    resultados = {}
    inicio = time.time()

    for k in [1, 3, 5, 7]:
        y_pred = prever(X_train, y_train, X_test, k)
        matriz = matriz_confusao(y_test, y_pred, n_classes)
        m = metricas(matriz)
        resultados[k] = {"matriz": matriz, **m}

        print(f"\n===== k = {k} (HARDCORE) =====")
        print(f"Acurácia: {m['acuracia']:.4f}")
        print(f"Precisão média: {m['precisao_media']:.4f}")
        print(f"Revocação média: {m['revocacao_media']:.4f}")
        print("Matriz de confusão:")
        print(matriz)

    tempo_total = time.time() - inicio
    print(f"\nTempo total de execução (4 valores de k): {tempo_total:.4f}s")

    return resultados, tempo_total


if __name__ == "__main__":
    executar()
