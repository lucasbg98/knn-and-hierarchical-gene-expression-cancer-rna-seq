# knn-and-hierarchical-gene-expression-cancer-rna-seq
Repositório referente a um trabalho para contextualizar o uso dos três modelos de Clustering: K-means, Hierárquico e DBSCAN, e fazer uma comparação com os resultados alcançados.

### Contextualização dos dados:
Nessa atividade, foi utilizada uma base de dados da UC Irvine contendo dados de
pacientes com câncer. A coleção de dados é uma parte da sequência RNA (Hi-Seq) PANCAN, ela
representa uma extração aleatória de expressões genéticas de pacientes tendo 5 diferentes tipos de
tumor, BRCA, KIRC, COAD, LUAD e PREAD.

O DataFrame possui dados de 801 pacientes, cada valor representando a quantidade do gene
presente no RNA do paciente.

###Desenvolvimento
Foi aplicado o pré-processamento nos dados do dataset:
- Verificação da existência de valores faltantes
- Normalização dos dados
- Aplicação de uma técnica de redução de dimensionalidade (como o PCA)

Após o pré-processamento, foi realizada a aplicação dos três modelos de clustering:
- K-Nearest Neighbors (KNN)
- Modelo de agrupamento Hierárquico
- DBSCAN

**Observação:**
  Por padrão os modelos de agrupamento exigem que seja informado como parâmetro o número de clusters que deve ser utilizado para realização dos calculos de agrupamento, porém neste trabalho foi utilizado o metodo silhouette. Através do silhouette_score
  foi possível definir o número de clusters ideal para cada modelo, tornando assim de forma dinâmica a seleção da quantidade de clusters necessária para cada modelo durante o calculo do agrupamento dos dados.

  ### Resultados
  Os resultados do agrupamento realizado por cada um dos modelos de clustering foram demonstrados atráves de gráficos comparativos para facilitação e entendimento.

