import numpy as np

def criar_matriz_jacobi(A, i, j, n):
    # Iniciaremos J como uma matriz indetidade nxn
    J = np.identity(n)
    
    # Usaremos para calcular o angulo e analizar caso A[i,j] < 10⁻6
    a_jj = A[j, j]
    a_ij = A[i, j]

    # Considerar Ai,j = 0, retorna a matriz identidade 
    if np.abs(a_ij) < 1e-6:
        return J

    # Calcula o cosseno e o seno do ângulo de rotação
    # Usamos uma fórmula numericamente estável
    r = np.hypot(a_jj, a_ij) # r = sqrt(a_jj**2 + a_ij**2)
    c = a_jj / r   # cos(theta)
    s = -a_ij / r  # sin(theta)

    # Preenche os quatro elementos da matriz de rotação
    J[j, j] = c
    J[i, i] = c
    J[i, j] = s   # No texto, sen(theta) está em J[i,j] para R = J*A
    J[j, i] = -s  # e -sen(theta) está em J[j,i]
    
    return J

# -----------------------------------------------------------------------------
# 3.1.2 Decomposição QR
# -----------------------------------------------------------------------------
def decomposicao_qr_jacobi(A, n):
    """
    Realiza a decomposição A = QR usando uma sequência de rotações de Jacobi.
    """
    R_atual = A.copy()
    
    # Q_T acumulará o produto de todas as matrizes de Jacobi J
    Q_T = np.identity(n)

    # Loop sobre as colunas para zerar os elementos abaixo da diagonal
    for j in range(n - 1):
        # Loop sobre as linhas abaixo da diagonal para a coluna j
        for i in range(j + 1, n):
            # Cria a matriz de Jacobi para zerar o elemento (i, j) de R_atual
            J = criar_matriz_jacobi(R_atual, i, j, n)
            
            # Aplica a rotação para zerar o elemento (i,j)
            # R_nova = J * R_atual
            R_atual = np.dot(J, R_atual)
            
            # Acumula a transformação em Q_T
            # Q_T_nova = J * Q_T_antiga
            Q_T = np.dot(J, Q_T)

    # Ao final, R_atual é a matriz triangular superior R
    R = R_atual
    # Q é a transposta da matriz acumulada Q_T
    Q = Q_T.T
    
    return Q, R

# -----------------------------------------------------------------------------
# 3.1.1 Algoritmo QR Principal
# -----------------------------------------------------------------------------
def metodo_qr(A, tol=1e-9, max_iter=1000):
    """
    Calcula os autovalores e autovetores de uma matriz A (preferencialmente simétrica)
    usando o algoritmo QR iterativo.
    """
    n = A.shape[0]
    
    # A_atual é a matriz que será transformada em cada iteração (A_k)
    A_atual = A.copy()
    
    # P acumulará o produto de todas as matrizes Q (Q_1 * Q_2 * ...)
    # No final, suas colunas serão os autovetores.
    P = np.identity(n)
    
    for k in range(max_iter):
        # Passo 1: Decomposição QR de A_atual
        Q, R = decomposicao_qr_jacobi(A_atual, n)
        
        # Passo 2: Calcular a nova matriz A_atual = R * Q
        A_nova = np.dot(R, Q)
        
        # Acumular o produto das matrizes Q para obter os autovetores
        P = np.dot(P, Q)
        
        # Critério de convergência:
        # Soma dos quadrados dos elementos abaixo da diagonal.
        # np.tril(A_nova, k=-1) pega todos os elementos abaixo da diagonal principal.
        soma_quadrados_off_diag = np.sum(np.square(np.tril(A_nova, k=-1)))
        
        # Atualiza A_atual para a próxima iteração
        A_atual = A_nova
        
        if soma_quadrados_off_diag < tol:
            print(f"Convergência alcançada na iteração {k+1}.")
            break
    else: # Executado se o loop for terminar sem 'break'
        print(f"Máximo de iterações ({max_iter}) atingido. A convergência pode não ter sido completa.")

    # Os autovalores são os elementos da diagonal da matriz final A_atual
    autovalores = np.diag(A_atual)
    
    # Os autovetores são as colunas da matriz P acumulada
    autovetores = P
    
    return autovetores, autovalores

# --- Exemplo de Uso ---
if __name__ == '__main__':
    # Definindo uma matriz simétrica de exemplo
    # A = np.array([[ 4.0,  1.0, -2.0,  2.0],
    #               [ 1.0,  2.0,  0.0,  1.0],
    #               [-2.0,  0.0,  3.0, -2.0],
    #               [ 2.0,  1.0, -2.0, -1.0]])

    A = np.array([[6., 2., 1.],
                  [2., 3., 1.],
                  [1., 1., 1.]])

    print("Matriz Original A:\n", A)
    print("-" * 30)

    # Executando o método QR
    autovetores_qr, autovalores_qr = metodo_qr(A)

    print("\nAutovalores encontrados (Método QR):\n", autovalores_qr)
    print("\nAutovetores encontrados (colunas da matriz P):\n", autovetores_qr)
    print("-" * 30)
    
    # Para comparação, vamos usar a função do numpy
    autovalores_np, autovetores_np = np.linalg.eigh(A) # eigh é para matrizes simétricas
    
    print("\nAutovalores (Numpy para comparação):\n", autovalores_np)
    print("\nAutovetores (Numpy para comparação):\n", autovetores_np)
    
    # Verificação: A * v = lambda * v
    print("\nVerificando o primeiro autovetor/autovalor (A @ v1):")
    v1 = autovetores_qr[:, 0]
    lambda1 = autovalores_qr[0]
    print(A @ v1)
    
    print("\nVerificando o primeiro autovetor/autovalor (lambda1 * v1):")
    print(lambda1 * v1)