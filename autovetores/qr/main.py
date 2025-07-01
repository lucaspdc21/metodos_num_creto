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

    theta = 0.0
    epsilon = 1e-6 

    # Caso especial: o denominador (a_jj) é zero.
    if np.abs(a_jj) <= epsilon:
        if a_ij < 0:
            # Se a_ij é negativo, -a_ij é positivo. tan(θ) -> +inf
            theta = np.pi / 2
        else:
            # Se a_ij é positivo, -a_ij é negativo. tan(θ) -> -inf
            theta = -np.pi / 2
    else:
        theta = np.arctan(-a_ij / a_jj)

    c = np.cos(theta)
    s = np.sin(theta)

    J[j, j] = c
    J[i, i] = c
    J[i, j] = s   
    J[j, i] = -s  
    
    return J

def decomposicao_qr_jacobi(A, n):
    R_atual = A.copy()
    Q_T = np.identity(n)

    for j in range(n - 1):
        for i in range(j + 1, n):
            J = criar_matriz_jacobi(R_atual, i, j, n)
            R_atual = np.dot(J, R_atual)
            Q_T = np.dot(J, Q_T)
    R = R_atual
    Q = Q_T.T
    
    return Q, R

def metodo_qr(A, tol=1e-6):

    n = A.shape[0]
    A_atual = A.copy()
    
    P = np.identity(n)
    k = 0
    while (True):
        Q, R = decomposicao_qr_jacobi(A_atual, n)
        A_nova = np.dot(R, Q)
        P = np.dot(P, Q)
        
        soma_quadrados_abaixo_diag = np.sum(np.square(np.tril(A_nova, k=-1)))

        A_atual = A_nova
        k += 1
        if soma_quadrados_abaixo_diag < tol:
            print(f"Convergência alcançada na iteração {k}.")
            break
    # Os autovalores são os elementos da diagonal da matriz final A_atual
    autovalores = np.diag(A_atual)
    
    # Os autovetores são as colunas da matriz P acumulada
    autovetores = P
    
    return autovetores, autovalores

if __name__ == '__main__':

    A = np.array([[40, 8, 4, 2, 1],
                  [8, 30, 12, 6, 2],
                  [4, 12, 20, 1, 2],
                  [2, 6, 1, 25, 4],
                  [1, 2, 2, 4, 5]], dtype=float)

    print("Matriz Original A:\n", A)

    autovetores_qr, autovalores_qr = metodo_qr(A)

    print("\nAutovalores encontrados :\n", autovalores_qr)
    print("\nAutovetores encontrados:\n", autovetores_qr)
    
    autovalores_np, autovetores_np = np.linalg.eigh(A) 
    
    print("\nAutovalores Corretos :\n", autovalores_np)
    print("\nAutovetores Corretos :\n", autovetores_np)
    