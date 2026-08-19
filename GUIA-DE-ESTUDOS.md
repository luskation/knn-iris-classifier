# Guia de Estudos — Trabalho Prático 01: Classificação KNN

Este guia foi pensado para você **construir o conhecimento**, não para copiar código.
Cada etapa tem: o conceito por trás, o que você precisa fazer, perguntas para você
se testar, e a sugestão de commit correspondente. Eu (Claude) vou te acompanhar
como um professor em plantão de dúvidas — você escreve o código e faz os commits.

## Convenção de commits

Prefixo em inglês (padrão [Conventional Commits](https://www.conventionalcommits.org/)),
mensagem em português:

| Prefixo | Quando usar |
|---|---|
| `feat:` | uma nova funcionalidade/etapa do algoritmo |
| `docs:` | documentação (README, guia, relatório) |
| `chore:` | organização, estrutura de pastas, configs |
| `refactor:` | melhorar código existente sem mudar comportamento |
| `test:` | testes ou validações |
| `fix:` | correção de bug |

Exemplo: `feat: implementa função de distância euclidiana`

**Regra de ouro:** se você consegue descrever o commit em uma frase só e ele faz
"uma coisa", está no tamanho certo. Evite commits gigantes tipo "implementa tudo".

---

## Etapa 0 — Estrutura do repositório

Crie a estrutura de pastas antes de escrever qualquer lógica.

```
knn-from-scratch-iris/
├── README.md
├── GUIA-DE-ESTUDOS.md
├── requirements.txt
├── src/
│   ├── knn_scratch.py      # sua implementação do zero
│   └── knn_sklearn.py      # versão com sklearn
├── notebook.ipynb          # (opcional) versão exploratória
├── relatorio.pdf
└── .gitignore
```

**Pergunta para você:** por que separar `knn_scratch.py` de `knn_sklearn.py` em vez
de colocar tudo num arquivo só? (Pense em organização e em facilitar a comparação depois.)

`chore: cria estrutura inicial do repositório`

---

## Etapa 1 — Carregar e preparar os dados

**Conceito:** o KNN é sensível à escala dos atributos, porque ele decide "vizinhança"
por distância. Se um atributo tiver valores muito maiores que os outros, ele domina
o cálculo da distância injustamente.

**O que fazer:**
1. Carregue a base Iris (`sklearn.datasets.load_iris()` é aceitável — é só leitura de dado, não é o algoritmo).
2. Separe `X` (atributos) e `y` (classes).
3. Divida em treino e teste (sugestão: 70/30 ou 80/20, com estratificação por classe).
4. Decida se vai normalizar os dados (ex: `StandardScaler` ou normalização manual) e **justifique isso no relatório**.

**Perguntas para você:**
- O que aconteceria com a distância euclidiana se um atributo estivesse em cm e outro em km?
- Por que é importante manter a **mesma divisão treino/teste** para as duas implementações (hardcore e sklearn)?

`feat: carrega dataset Iris e realiza divisão treino/teste`

---

## Etapa 2 — Função de distância

**Conceito:** a distância euclidiana entre dois pontos `p` e `q` em n dimensões é:

```
d(p, q) = sqrt( soma( (p_i - q_i)^2 ) )   para i = 1..n
```

**O que fazer:** implemente essa função recebendo dois vetores de atributos e
retornando um número (a distância).

**Pergunta para você:** dá para calcular isso sem nenhum laço explícito, usando
operações vetorizadas do NumPy? Por que isso seria mais rápido?

`feat: implementa função de distância euclidiana`

---

## Etapa 3 — Encontrar os k vizinhos mais próximos

**Conceito:** para classificar um ponto novo, você calcula a distância dele até
**todos** os pontos de treino, e pega os `k` com menor distância.

**O que fazer:**
1. Para um ponto de teste, calcule a distância até cada ponto de treino.
2. Ordene essas distâncias.
3. Selecione os índices dos `k` menores.

**Pergunta para você:** o que você guarda junto com a distância — só o índice, ou
também a classe do vizinho? Por que isso importa para a próxima etapa?

`feat: implementa busca dos k vizinhos mais próximos`

---

## Etapa 4 — Votação majoritária (com desempate)

**Conceito:** depois de achar os `k` vizinhos, a classe prevista é a que aparece
com mais frequência entre eles.

**O que fazer:**
1. Conte quantos vizinhos pertencem a cada classe.
2. Retorne a classe mais frequente.
3. **Trate o empate** — por exemplo, quando k=4 e dá 2 votos para cada classe.
   Uma estratégia comum é desempatar pegando a classe do vizinho mais próximo
   entre os empatados.

**Pergunta para você:** por que os valores de k pedidos no enunciado (1, 3, 5, 7)
são todos ímpares? Isso evita algum tipo de empate quando o problema tem 2 classes?
E se o problema (como a Iris) tem **3** classes, ímpar ainda garante ausência de empate?

`feat: implementa votação majoritária com desempate`

---

## Etapa 5 — Função de predição completa

**O que fazer:** junte as etapas 2, 3 e 4 numa função `predict(X_treino, y_treino, X_novo, k)`
que retorna a classe prevista para um ou mais pontos novos.

**Pergunta para você:** sua função consegue prever para **um conjunto** de pontos
de teste de uma vez (não só um ponto por vez)? Como você organizaria isso —
laço externo sobre os pontos de teste, chamando a lógica das etapas 2-4 para cada um?

`feat: implementa função de predição do KNN hardcore`

---

## Etapa 6 — Avaliar para k = {1, 3, 5, 7}

**O que fazer:**
1. Rode sua predição no conjunto de teste para cada valor de k.
2. Calcule a taxa de acerto (acurácia) = acertos / total.
3. Guarde os resultados (uma tabela ou dicionário `{k: acurácia}`) para usar no relatório.

**Pergunta para você:** o que você espera que aconteça com a acurácia à medida que
k aumenta muito (ex: k=50, numa base de ~100 exemplos de treino)? Por que isso acontece?

`feat: avalia KNN hardcore para k=1,3,5,7`

---

## Etapa 7 — Matriz de confusão e métricas (versão hardcore)

**Conceito:**
- **Matriz de confusão**: linhas = classe real, colunas = classe prevista. A diagonal são os acertos.
- **Precisão** (por classe): dos que o modelo previu como classe X, quantos realmente eram X?
- **Revocação/recall** (por classe): dos que realmente eram classe X, quantos o modelo acertou?
- **Acurácia** (geral): total de acertos / total de exemplos.

**O que fazer:** você pode construir a matriz de confusão manualmente (contagem
com um laço) ou usar `sklearn.metrics.confusion_matrix` **apenas para exibir/plotar**
— isso é uma ferramenta de avaliação, não o classificador em si, então não fere a regra do trabalho.

**Pergunta para você:** olhando a matriz de confusão da sua implementação, existe
alguma confusão sistemática entre duas classes específicas da Iris? Faz sentido
biologicamente (alguma espécie mais parecida com outra)?

`feat: calcula matriz de confusão e métricas do KNN hardcore`

---

## Etapa 8 — Implementação com Sklearn

**O que fazer:**
1. Use `KNeighborsClassifier` com o **mesmo split** de treino/teste da etapa 1.
2. Rode para os mesmos valores de k = {1, 3, 5, 7}.
3. Gere matriz de confusão e métricas (aqui pode usar as funções prontas do sklearn sem culpa).

**Pergunta para você:** quais parâmetros do `KNeighborsClassifier` poderiam mudar
o resultado além do `n_neighbors`? (dica: `weights`, `metric`)

`feat: implementa classificador KNN com sklearn`
`feat: calcula matriz de confusão e métricas do KNN sklearn`

---

## Etapa 9 — Comparação de desempenho

**O que fazer:**
1. Monte uma tabela comparando, para cada k, a acurácia (e opcionalmente precisão/revocação) das duas implementações.
2. Opcional: meça o tempo de execução de cada uma (`time.time()` antes/depois).
3. Escreva a análise: elas convergem para resultados parecidos? Isso é esperado, já que ambas implementam o mesmo algoritmo — a diferença deveria estar em performance/otimização, não em qualidade do resultado.

**Pergunta para você:** se as métricas forem muito diferentes entre as duas
implementações, isso é sinal de quê? (Provavelmente um bug na versão hardcore —
é um bom sinal de alerta para revisar.)

`feat: compara desempenho entre implementação hardcore e sklearn`

---

## Etapa 10 — Relatório (PDF, até 1 página)

**O que incluir:**
- Tabela comparativa de métricas (precisão, revocação, acurácia) para os dois classificadores, nos 4 valores de k.
- Breve análise: os resultados fazem sentido? Houve diferença de tempo de execução?
- Conclusão: o que esse exercício te ensinou sobre como o KNN funciona por dentro?

`docs: adiciona relatório em PDF com comparação e conclusão`

---

## Etapa 11 — Finalização

1. Revise o README com os resultados finais e instruções de como rodar o projeto.
2. Confira se o `.py`/`.ipynb` roda do zero sem erros.
3. Gere o `.zip` no formato `nome1_nome2.zip` pedido no enunciado.

`docs: atualiza README com resultados finais`
`chore: organiza entrega final`

---

## Checklist final

- [ ] Estrutura de pastas criada (`chore: cria estrutura inicial`)
- [ ] Dataset Iris carregado e dividido em treino/teste (`feat`)
- [ ] Função de distância euclidiana implementada (`feat`)
- [ ] Busca dos k vizinhos mais próximos implementada (`feat`)
- [ ] Votação majoritária com tratamento de empate implementada (`feat`)
- [ ] Função de predição completa (KNN hardcore) implementada (`feat`)
- [ ] Avaliação para k = {1, 3, 5, 7} feita e taxa de reconhecimento registrada (`feat`)
- [ ] Matriz de confusão e métricas (precisão, revocação, acurácia) da versão hardcore (`feat`)
- [ ] Classificador com Sklearn implementado, mesmo split e mesmos k (`feat`)
- [ ] Matriz de confusão e métricas da versão Sklearn (`feat`)
- [ ] Comparação entre as duas implementações (métricas e, se possível, tempo) (`feat`)
- [ ] Relatório PDF de até 1 página com comparação e conclusão (`docs`)
- [ ] README atualizado com resultados finais (`docs`)
- [ ] Entrega organizada no formato `nome1_nome2.zip` (`chore`)

Quando terminar cada item, volte aqui e me chame para revisarmos juntos antes do
próximo passo — é assim que vamos garantir que você realmente entendeu cada parte,
não só "fez funcionar".
