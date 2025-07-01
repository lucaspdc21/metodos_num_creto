import numpy as np
from scipy.linalg import lu_factor, lu_solve

def metodo_da_potencia_regular(A, chute_inicial, tolerancia=1e-6, max_iteracoes=1000):

    # Passos de inicialização.
    matriz_A = np.array(A, dtype=float)
    x_k = np.array(chute_inicial, dtype=float)
    
    lambda_novo = 0.0
    iteracao_atual = 0

    # Critério de parada: número máximo de iterações
    while iteracao_atual < max_iteracoes:
        # Salvando o autovalor da iteração anterior para calcular o erro relativo
        lambda_antigo = lambda_novo

        # Devemos normalizar o vetor a cada iteração para evitar valores muito grandes ou muito pequenos, não estamos alterando sua direção.
        x_k_normalizado = x_k / np.linalg.norm(x_k)

        # Multiplicando o vetor k pela matriz para obter o vetor k+1
        v_novo = np.dot(matriz_A, x_k_normalizado)

        # Quociente de Rayleigh
        lambda_novo = np.dot(x_k_normalizado.T, v_novo)
        
        # Atualizar x_k para a procima iteração
        x_k = v_novo

        # Caluclando o erro relativo
        if lambda_novo != 0:
            erro = np.abs((lambda_novo - lambda_antigo) / lambda_novo)
            if erro < tolerancia:
                print(f"Convergência alcançada na iteração {iteracao_atual + 1}.")
                print(f'Parada por erro relativo: {erro:.6f} < {tolerancia}')
                break
        
        iteracao_atual += 1

    autovetor_dominante = x_k
    
    return lambda_novo, autovetor_dominante

def metodo_da_potencia_inverso(A, chute_inicial, tolerancia=1e-6, max_iteracoes=1000):
    # Vou modificar o método anterior para calcular o autovalor dominante de A inversa
    matriz_A = np.array(A, dtype=float)
    x_k = np.array(chute_inicial, dtype=float)

    lambda_novo = 0.0
    iteracao_atual = 0

    # Fzendo a decomposição LU da matriz A)
    lu, piv = lu_factor(matriz_A)

    while iteracao_atual < max_iteracoes:
        lambda_antigo = lambda_novo
        
        x_k_normalizado = x_k / np.linalg.norm(x_k)

        # Step 8: Resolver A * v_novo = x_k_normalizado usando a decomposição LU.
        # Isto é equivalente a v_novo = A_inversa * x_k_normalizado
        v_novo = lu_solve((lu, piv), x_k_normalizado)
        
        # Step 9: Calcular a nova estimativa do autovalor de A_inversa
        lambda_inv_novo = np.dot(x_k_normalizado.T, v_novo)
        
        # Atualizar x_k para a próxima iteração
        x_k = v_novo

        # Step 10: Verificar convergência
        if lambda_novo != 0:
            erro = np.abs((lambda_novo - lambda_antigo) / lambda_novo)
            if erro < tolerancia:
                print(f"Convergência do Método Inverso alcançada na iteração {iteracao_atual + 1}.")
                print(f'Parada por erro relativo: {erro:.6f} < {tolerancia}')
                break
        
        iteracao_atual += 1

    # Step 11: Calcular o autovalor de A (o inverso do resultado do loop)
    autovalor_menor = 1 / lambda_inv_novo
    
    # Step 12: O autovetor correspondente
    autovetor_menor = x_k
    
    return autovalor_menor, autovetor_menor

def potencia_com_deslocamento(A, chute_inicial, mu, tolerancia=1e-6, max_iteracoes=1000):
    matriz_A = np.array(A, dtype=float)
    n = matriz_A.shape[0]
    
    matriz_identidade = np.identity(n)
    A_deslocada = matriz_A - mu * matriz_identidade
    
    lambda_a, x_c = metodo_da_potencia_inverso(A_deslocada, chute_inicial, tolerancia, max_iteracoes)
    
    if lambda_a is None:
        return None, None

    lambda_i = lambda_a + mu
    
    x_i = x_c
    
    return lambda_i, x_i



if __name__ == "__main__":
        
    A1 = [[5, 2, 1], 
          [2, 3, 1], 
          [1, 1, 2]]
          
    v0 = [1, 1, 1]

    autovalores_reais = np.linalg.eigvals(A1)
    print(f"Autovalores reais da Matriz A1 (calculados com NumPy): {np.sort(autovalores_reais)}")
    print("-" * 50)

    mu1 = 1.2
    print(f"Buscando o autovalor mais próximo de μ = {mu1}...")
    lambda_prox1, autovetor_prox1 = potencia_com_deslocamento(A1, v0, mu=mu1)
    
    if lambda_prox1 is not None:
        print(f"Autovalor Encontrado (λᵢ): {lambda_prox1:.6f}")
        print(f"Autovetor Correspondente (xᵢ): {autovetor_prox1}")
        # Verificação: A * xᵢ ≈ λᵢ * xᵢ
        verificacao_Ax = np.dot(A1, autovetor_prox1)
        verificacao_lambdax = lambda_prox1 * autovetor_prox1
        print(f"Verificação (A*xᵢ): {verificacao_Ax}")
        print(f"Verificação (λᵢ*xᵢ): {verificacao_lambdax}\n")

    mu2 = 6.0
    print(f"Buscando o autovalor mais próximo de μ = {mu2}...")
    lambda_prox2, autovetor_prox2 = potencia_com_deslocamento(A1, v0, mu=mu2)

    if lambda_prox2 is not None:
        print(f"Autovalor Encontrado (λᵢ): {lambda_prox2:.6f}")
        print(f"Autovetor Correspondente (xᵢ): {autovetor_prox2}")
        verificacao_Ax = np.dot(A1, autovetor_prox2)
        verificacao_lambdax = lambda_prox2 * autovetor_prox2
        print(f"Verificação (A*xᵢ): {verificacao_Ax}")
        print(f"Verificação (λᵢ*xᵢ): {verificacao_lambdax}")
    
    A1 = [[5, 2, 1], 
          [2, 3, 1], 
          [1, 1, 2]]
          
    v0_A1 = [1, 1, 1]
    
    lambda_1, autovetor_1 = metodo_da_potencia_regular(A1, v0_A1)
    
    print("\nA1:")
    print(f"Autovalor Dominante (λ₁): {lambda_1:.6f}")
    print(f"Autovetor Correspondente (x₁): {autovetor_1}")
    # Verificando A * v = λ * v 
    print(f"Verificação: {np.dot(A1, autovetor_1)}")
    print(f"Verificação: {lambda_1 * autovetor_1}\n\n")

    lambda_n, autovetor_n = metodo_da_potencia_inverso(A1, v0_A1)

    A2 = [[40, 8, 4, 2, 1], 
          [8, 30, 12, 6, 2], 
          [4, 12, 20, 1, 2], 
          [2, 6, 1, 25, 4], 
          [1, 2, 2, 4, 5]]
          
    v0_A2 = [1, 1, 1, 1, 1]

    lambda_2, autovetor_2 = metodo_da_potencia_regular(A2, v0_A2, max_iteracoes=1000)
    
    print("\nA2:")
    print(f"Autovalor Dominante: {lambda_2:.6f}")
    print(f"Autovetor Correspondente: {autovetor_2}")
    # Verificando A * v = λ * v
    print(f"Verificação (A * x): {np.dot(A2, autovetor_2)}")
    print(f"Verificação (λ * x): {lambda_2 * autovetor_2}")
