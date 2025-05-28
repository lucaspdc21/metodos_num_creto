#include <iostream>   
#include <vector>     
#include <cmath>      
#include <functional> 
#include <iomanip>    
#include "integracao_metodos.cpp" 

double funcao_simples(double x) {
    return pow(x, 3) + 2 * pow(x, 2) + 5;
}


int main() {
    // std::function<double(double)> f = funcao_exemplo;
    std::function<double(double)> f = funcao_simples;

    // Parâmetros de integração
    double a = 0.0; // Limite inferior
    double b = 2.0; // Limite superior

    // Número de subintervalos (precisa ser par E múltiplo de 3 para todos funcionarem)
    int N = 60;

    // --- Chamada e Teste dos Métodos ---
    std::cout << "Calculando a integral de f(x) de " << a << " a " << b << " com N = " << N << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    // Configura a precisão da saída
    std::cout << std::fixed << std::setprecision(10);

    // Regra do Trapézio
    double resultado_trapezio = regra_trapezio(f, a, b, N);
    std::cout << "Regra do Trapézio:     " << resultado_trapezio << std::endl;

    // Regra 1/3 de Simpson
    if (N % 2 == 0) {
        double resultado_simpson13 = regra_simpson_1_3(f, a, b, N);
        std::cout << "Regra 1/3 de Simpson:  " << resultado_simpson13 << std::endl;
    } else {
        std::cout << "Regra 1/3 de Simpson: N deve ser par!" << std::endl;
    }

    // Regra 3/8 de Simpson
    if (N % 3 == 0) {
        double resultado_simpson38 = regra_simpson_3_8(f, a, b, N);
        std::cout << "Regra 3/8 de Simpson:  " << resultado_simpson38 << std::endl;
    } else {
        std::cout << "Regra 3/8 de Simpson: N deve ser múltiplo de 3!" << std::endl;
    }

    // Gauss-Legendre 2 Pontos
    double resultado_gl2 = gauss_legendre_2_pontos(f, a, b);
    std::cout << "Gauss-Legendre 2 pts:  " << resultado_gl2 << std::endl;

    // Gauss-Legendre 3 Pontos
    double resultado_gl3 = gauss_legendre_3_pontos(f, a, b);
    std::cout << "Gauss-Legendre 3 pts:  " << resultado_gl3 << std::endl;

    // Gauss-Legendre 4 Pontos
    double resultado_gl4 = gauss_legendre_4_pontos(f, a, b);
    std::cout << "Gauss-Legendre 4 pts:  " << resultado_gl4 << std::endl;

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "(Valor exato para f(x)=x^3+2x^2+5 de 0 a 2 é ~19.3333333333)" << std::endl;

    return 0;
}

