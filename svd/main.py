import numpy as np

def svd_teorico(A):
    m, n = A.shape

    B = A.T @ A
    
    # Calcular os autovalores e autovetores de B = A.T @ A
    autovalores, V = np.linalg.eig(B)

    # Ordenando os autovalores e autovetores
    idx = autovalores.argsort()[::-1]
    autovalores = autovalores[idx]
    V = V[:, idx]

    # Calculando os valores singulares, valores = raiz quadrada dos autovalores
    valores_singulares = np.sqrt(np.abs(autovalores))

    # Salvando os v transposto como Vh
    Vh = V.T


    # Usando a relação u_i = (1/sigma_i) * A @ v_i
    tolerancia = 1e-10
    k = np.sum(valores_singulares > tolerancia) # Número de valores singulares não nulos

    Sigmas = valores_singulares[:k]
    Vh = Vh[:k, :] # Redimensiona para ficar economico
    
    U = np.zeros((m, k))

    for i in range(k):
        U[:, i] = (1 / Sigmas[i]) * (A @ V[:, i])

    return U, Sigmas, Vh

def main():
    np.set_printoptions(precision=3, suppress=True)
    A = np.array([
        [0, 1, 1],
        [np.sqrt(2), 2, 0],
        [0, 1, 1],
    ])

    print("Matriz Original:\n", A)

    U_teo, Sigmas_teo, Vh_teo = svd_teorico(A)

    print("U (teórico):\n", U_teo)
    print("\nS (teórico) - valores singulares:\n", Sigmas_teo)
    print("\nVh (teórico):\n", Vh_teo)

    print("\n\n--- Verificando U @ S @ Vh = A ---")

    Sigma_matriz = np.zeros_like(A, dtype=float)
    Sigma_matriz[:len(Sigmas_teo), :len(Sigmas_teo)] = np.diag(Sigmas_teo)
    A_reconstruida = U_teo @ Sigma_matriz @ Vh_teo

    print("\nMatriz A Reconstruída (U @ Sigma @ Vh):\n", A_reconstruida)


if __name__ == "__main__":
    main()