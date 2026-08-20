# KNN Iris — Hardcore vs Scikit-learn

Implementação do algoritmo **K-Nearest Neighbors (KNN)** do zero, comparada
com a implementação de referência do Scikit-learn, aplicada à base de dados
**Iris** (150 amostras, 4 atributos, 3 classes: *setosa*, *versicolor*,
*virginica*).

O objetivo é validar, na prática, que uma implementação manual do KNN
(distância euclidiana, busca de vizinhos, votação majoritária com
desempate, matriz de confusão e métricas calculadas à mão) produz
resultados equivalentes aos do `KNeighborsClassifier`.

## Estrutura do projeto

```
knn-iris/
├── src/
│   ├── knn_scratch.py     # KNN implementado do zero
│   └── knn_sklearn.py     # KNN com scikit-learn (baseline de comparação)
├── requirements.txt
├── MANUAL-KNN-IRIS.md     # detalhamento técnico e resultados completos
├── relatorio.pdf          # relatório de 1 página (entrega acadêmica)
└── README.md
```

## Requisitos

```
numpy
pandas
scikit-learn
matplotlib
jupyter
```

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
pip install -r requirements.txt

python src/knn_scratch.py        # implementação hardcore
python src/knn_sklearn.py        # implementação sklearn (comparação)
```

Ambos os scripts usam o mesmo split treino/teste (70/30, estratificado,
`random_state=42`) e os mesmos dados normalizados (`StandardScaler`), para
garantir uma comparação justa.

## Metodologia

1. **Carregamento e preparação** — dataset Iris via `sklearn.datasets`,
   split 70/30 estratificado por classe, normalização com `StandardScaler`.
2. **Distância euclidiana** — `sqrt(soma((p_i - q_i)^2))`.
3. **Busca dos k vizinhos mais próximos** — distância do ponto de teste a
   todos os pontos de treino, ordenação e seleção dos `k` menores.
4. **Votação majoritária** — classe mais frequente entre os `k` vizinhos,
   com desempate pelo vizinho mais próximo entre as classes empatadas.
5. **Avaliação** — matriz de confusão e métricas (acurácia, precisão e
   revocação por classe) calculadas manualmente, para `k ∈ {1, 3, 5, 7}`.
6. **Comparação** — os mesmos `k` e o mesmo split são rodados com
   `KNeighborsClassifier` para validar a implementação hardcore.

## Resultados

| k | Acurácia (hardcore) | Precisão média | Revocação média | Acurácia (sklearn) | Precisão média | Revocação média |
|---|---|---|---|---|---|---|
| 1 | 0.9333 | 0.9444 | 0.9333 | 0.9333 | 0.9444 | 0.9333 |
| 3 | 0.9111 | 0.9298 | 0.9111 | 0.9111 | 0.9298 | 0.9111 |
| 5 | 0.9111 | 0.9298 | 0.9111 | 0.9111 | 0.9298 | 0.9111 |
| 7 | 0.9333 | 0.9444 | 0.9333 | 0.9333 | 0.9444 | 0.9333 |

As métricas são **idênticas** entre as duas implementações em todos os
valores de `k`, confirmando que a lógica hardcore está correta. O único
erro sistemático ocorre entre *versicolor* e *virginica* — nunca com
*setosa*, que é linearmente separável das outras duas —, o que é esperado
dada a sobreposição natural de medidas de pétala/sépala entre essas duas
espécies.

Detalhes completos (matrizes de confusão, tempo de execução, análise) em
[`MANUAL-KNN-IRIS.md`](MANUAL-KNN-IRIS.md) e [`relatorio.pdf`](relatorio.pdf).

## Conclusão

As duas implementações produzem resultados equivalentes em qualidade
(acurácia, precisão, revocação). A vantagem da biblioteca está na
eficiência computacional (estruturas como KD-Tree/Ball-Tree), não na
qualidade da predição — o algoritmo em si é o mesmo.
