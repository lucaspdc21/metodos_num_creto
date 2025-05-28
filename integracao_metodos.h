#ifndef INTEGRACAO_H
#define INTEGRACAO_H

#include <functional> 

/**
 * @brief Calcula a integral usando a Regra Composta do Trapézio.
 * @param func A função f(x) a ser integrada.
 * @param a Limite inferior de integração.
 * @param b Limite superior de integração.
 * @param N Número de subintervalos (deve ser >= 1).
 * @return O valor aproximado da integral.
 */
double regra_trapezio(const std::function<double(double)>& func, double a, double b, int N);

/**
 * @brief Calcula a integral usando a Regra Composta 1/3 de Simpson.
 * @param func A função f(x) a ser integrada.
 * @param a Limite inferior de integração.
 * @param b Limite superior de integração.
 * @param N Número de subintervalos (DEVE SER PAR e >= 2).
 * @return O valor aproximado da integral.
 */
double regra_simpson_1_3(const std::function<double(double)>& func, double a, double b, int N);

/**
 * @brief Calcula a integral usando a Regra Composta 3/8 de Simpson.
 * @param func A função f(x) a ser integrada.
 * @param a Limite inferior de integração.
 * @param b Limite superior de integração.
 * @param N Número de subintervalos (DEVE SER MÚLTIPLO DE 3 e >= 3).
 * @return O valor aproximado da integral.
 */
double regra_simpson_3_8(const std::function<double(double)>& func, double a, double b, int N);

double gauss_legendre_2_pontos(const std::function<double(double)>& func, double a, double b);

/**
 * @brief Calcula a integral usando a Quadratura de Gauss-Legendre com 3 pontos.
 * @param func A função f(x) a ser integrada.
 * @param a Limite inferior de integração.
 * @param b Limite superior de integração.
 * @return O valor aproximado da integral.
 */
double gauss_legendre_3_pontos(const std::function<double(double)>& func, double a, double b);

/**
 * @brief Calcula a integral usando a Quadratura de Gauss-Legendre com 4 pontos.
 * @param func A função f(x) a ser integrada.
 * @param a Limite inferior de integração.
 * @param b Limite superior de integração.
 * @return O valor aproximado da integral.
 */
double gauss_legendre_4_pontos(const std::function<double(double)>& func, double a, double b);
#endif // INTEGRACAO_H