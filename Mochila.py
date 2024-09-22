import random

# Função para calcular o valor total, peso total e fitness de um cromossomo
def calcula_fitness(cromossomo, pesos_e_valores, peso_maximo):
    valor_total = 0
    peso_total = 0
    for i, gene in enumerate(cromossomo):
        if gene == 1:
            peso_total += pesos_e_valores[i][0]
            valor_total += pesos_e_valores[i][1]
    # Penalização se o peso ultrapassar o limite
    if peso_total > peso_maximo:
        return 0, peso_total  # Penaliza o fitness se o peso for excedido
    return valor_total, peso_total

# Função para criar um indivíduo (cromossomo)
def cria_individuo(tamanho):
    return [random.randint(0, 1) for _ in range(tamanho)]

# Função de crossover
def crossover(pai1, pai2):
    ponto_de_corte = random.randint(1, len(pai1) - 1)
    filho1 = pai1[:ponto_de_corte] + pai2[ponto_de_corte:]
    filho2 = pai2[:ponto_de_corte] + pai1[ponto_de_corte:]
    return filho1, filho2

# Função de mutação
def mutacao(cromossomo, taxa_mutacao=0.01):
    for i in range(len(cromossomo)):
        if random.random() < taxa_mutacao:
            cromossomo[i] = 1 - cromossomo[i]  # Inverte o gene
    return cromossomo

# Função principal do algoritmo genético
def algoritmo_genetico(pesos_e_valores, peso_maximo, num_cromossomos, geracoes):
    populacao = [cria_individuo(len(pesos_e_valores)) for _ in range(num_cromossomos)]
    melhor_individuos_por_geracao = []

    for geracao in range(geracoes):
        # Avaliar todos os indivíduos na população
        fitness_populacao = []
        for individuo in populacao:
            fitness, peso_total = calcula_fitness(individuo, pesos_e_valores, peso_maximo)
            fitness_populacao.append((fitness, peso_total, individuo))

        fitness_populacao.sort(reverse=True, key=lambda x: x[0])  # Ordena por fitness

        # Guardar o melhor indivíduo desta geração
        melhor_individuo = fitness_populacao[0]
        fitness, peso_total, cromossomo = melhor_individuo

        # Adicionar o valor total (fitness) e o cromossomo ao resultado
        melhor_individuos_por_geracao.append([fitness, cromossomo])

        # Seleção: selecionar os melhores indivíduos para crossover
        nova_populacao = []
        for _ in range(num_cromossomos // 2):
            pai1 = random.choice(fitness_populacao)[2]
            pai2 = random.choice(fitness_populacao)[2]
            filho1, filho2 = crossover(pai1, pai2)
            nova_populacao.append(mutacao(filho1))
            nova_populacao.append(mutacao(filho2))

        populacao = nova_populacao

    # Retorna os melhores indivíduos por geração
    return melhor_individuos_por_geracao

# Exemplo de uso
pesos_e_valores = [
    [2, 10], [4, 30], [6, 300], [8, 10], [8, 30], [8, 300],
    [12, 50], [25, 75], [50, 100], [100, 400]
]
peso_maximo = 100
numero_de_cromossomos = 150
geracoes = 50

resultado = algoritmo_genetico(pesos_e_valores, peso_maximo, numero_de_cromossomos, geracoes)

# Exibir o resultado em formato de array de arrays (valor total e cromossomo)
print(resultado)