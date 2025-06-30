#include "integracao_metodos.h"
#include <cmath>
#include <vector>
#include <stdexcept> 

double regra_trapezio(const std::function<double(double)>& func, double a, double b, int N) {
    if (N < 1) {
        throw std::invalid_argument("N deve ser pelo menos 1 para a Regra do Trapézio.");
    }

    double h = (b - a) / N;
    double soma = func(a) + func(b);

    for (int i = 1; i < N; ++i) {
        double x_i = a + i * h;
        soma += 2.0 * func(x_i);
    }

    return (h / 2.0) * soma;
}

double regra_simpson_1_3(const std::function<double(double)>& func, double a, double b, int N) {
    if (N < 2 || N % 2 != 0) {
        throw std::invalid_argument("N deve ser par e pelo menos 2 para a Regra 1/3 de Simpson.");
    }

    double h = (b - a) / N;
    double soma = func(a) + func(b); 

    for (int i = 1; i < N; i += 2) {
        double x_i = a + i * h;
        soma += 4.0 * func(x_i);
    }

    for (int i = 2; i < N - 1; i += 2) {
        double x_i = a + i * h;
        soma += 2.0 * func(x_i);
    }

    return (h / 3.0) * soma;
}

double regra_simpson_3_8(const std::function<double(double)>& func, double a, double b, int N) {
    if (N < 3 || N % 3 != 0) {
        throw std::invalid_argument("N deve ser múltiplo de 3 e pelo menos 3 para a Regra 3/8 de Simpson.");
    }

    double h = (b - a) / N;
    double soma = func(a) + func(b); 

    for (int i = 1; i < N; ++i) {
        double x_i = a + i * h;
        if (i % 3 == 0) {
            soma += 2.0 * func(x_i); 
        } else {
            soma += 3.0 * func(x_i);
        }
    }

    return (3.0 * h / 8.0) * soma;
}

/**
 * @brief Função auxiliar para a mudança de variável de [a, b] para [-1, 1].
 * @param alpha Ponto no intervalo [-1, 1].
 * @param a Limite inferior original.
 * @param b Limite superior original.
 * @return O ponto correspondente no intervalo [a, b].
 */
inline double transforma_x(double alpha, double a, double b) {
    return ((b + a) / 2.0) + ((b - a) / 2.0) * alpha;
}


double gauss_legendre_2_pontos(const std::function<double(double)>& func, double a, double b) {
    const double alpha1 = -1.0 / sqrt(3.0);
    const double alpha2 = +1.0 / sqrt(3.0);
    const double w1 = 1.0;
    const double w2 = 1.0;

    // Mapeia as raízes para o intervalo [a, b]
    double x1 = transforma_x(alpha1, a, b);
    double x2 = transforma_x(alpha2, a, b);

    // Calcula a soma ponderada
    double soma = w1 * func(x1) + w2 * func(x2);

    // Aplica o fator de escala
    return ((b - a) / 2.0) * soma;
}

double gauss_legendre_3_pontos(const std::function<double(double)>& func, double a, double b) {
    const double alpha1 = -sqrt(3.0 / 5.0); 
    const double alpha2 = 0.0;
    const double alpha3 = +sqrt(3.0 / 5.0); 
    const double w1 = 5.0 / 9.0;
    const double w2 = 8.0 / 9.0;
    const double w3 = 5.0 / 9.0;

    // Mapeia as raízes para o intervalo [a, b]
    double x1 = transforma_x(alpha1, a, b);
    double x2 = transforma_x(alpha2, a, b);
    double x3 = transforma_x(alpha3, a, b);

    // Calcula a soma ponderada
    double soma = w1 * func(x1) + w2 * func(x2) + w3 * func(x3);

    // Aplica o fator de escala
    return ((b - a) / 2.0) * soma;
}

double gauss_legendre_4_pontos(const std::function<double(double)>& func, double a, double b) {

    const double alpha1 = -0.8611363116;
    const double alpha2 = -0.3399810436;
    const double alpha3 = +0.3399810436;
    const double alpha4 = +0.8611363116;

    const double w1 = 0.3478548451;
    const double w2 = 0.6521451549;
    const double w3 = 0.6521451549;
    const double w4 = 0.3478548451;

    // Mapeia as raízes para o intervalo [a, b]
    double x1 = transforma_x(alpha1, a, b);
    double x2 = transforma_x(alpha2, a, b);
    double x3 = transforma_x(alpha3, a, b);
    double x4 = transforma_x(alpha4, a, b);

    // Calcula a soma ponderada
    double soma = w1 * func(x1) + w2 * func(x2) + w3 * func(x3) + w4 * func(x4);

    // Aplica o fator de escala
    return ((b - a) / 2.0) * soma;
}