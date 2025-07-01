import numpy as np

np.set_printoptions(precision=3, suppress=True)

def construir_matriz_householder_passo_i(A_passo_anterior, i, n):
    # Olhamos para a coluna i+1 que queremos zerar, em relação a matriz do passo anterior.
    w_sub = A_passo_anterior[i+1:, i].copy()
    
    # Se o vetor já for nulo, não é necessário fazer nada.
    # A matriz de Householder seria a identidade.
    if np.linalg.norm(w_sub) == 0:
        return np.eye(n)

    # Vetor w completo, como no pseudocódigo, para manter a lógica
    w = np.zeros(n)
    w[i+1:] = w_sub

    # Calcular o comprimento do vetor w (norma de w_sub)
    Lw = np.linalg.norm(w)

    # Criar o vetor w_linha (w')
    w_linha = np.zeros(n)
    sinal_oposto = 1 if w[i+1] < 0 else -1
    w_linha[i+1] = sinal_oposto * Lw

    # Calcular o vetor N
    N = w - w_linha
    
    # Normalizar o vetor N
    norm_N = np.linalg.norm(N)
    #if norm_N == 0:
    #    return np.eye(n)
    n_normalizado = N / norm_N

    # Montar a matriz de Householder H = I - 2nn^T
    H = np.eye(n) - 2 * np.outer(n_normalizado, n_normalizado)
    return H

def metodo_de_householder(A):
        
    n = A.shape[0]
    A_linha = A.copy()
    H_acumulada = np.eye(n)
    
    for i in range(n - 2):
        H_i = construir_matriz_householder_passo_i(A_linha, i, n)
        
        A_linha = H_i @ A_linha @ H_i 
        
        H_acumulada = H_acumulada @ H_i

    print("\nFim do Método de Householder\n")

    print(f"Matriz Acumulada H:\n{H_acumulada}\n")
    return A_linha, H_acumulada


def main():
    A = np.array([
        [40, 8, 4, 2, 1],
        [8, 30, 12, 6, 2],
        [4, 12, 20, 1, 2],
        [2, 6, 1, 25, 4],
        [1, 2, 2, 4, 5]
    ], dtype=float)

    print("Matriz Original A:")
    print(A)
    
    A_tridiagonal, H_acumulada = metodo_de_householder(A)
    
    print("Matriz Tridiagonal:")
    print(A_tridiagonal)
    print("Matriz Acumulada:")
    print(H_acumulada)

    print("Verificar ortoganalidade, deve ser identidade")
    identidade_calculada = H_acumulada.T @ H_acumulada 
    print(identidade_calculada)

    print("Verificar similiariadede")
    A_reconstruida = H_acumulada @ A_tridiagonal @ H_acumulada.T
    print(A_reconstruida)


if __name__ == "__main__":
    main()