#ifndef PROJET_ANNUEL_3IABD_LIB_H
#define PROJET_ANNUEL_3IABD_LIB_H

#include "../struct/pmc_struct.h"

PMC* init_pmc(const int* layers_sizes, int nb_layers, double learning_rate);

double act_sigmoid(double a);
double act_sigmoid_derivative(double sigmoid_res);

void propagation(const PMC* pmc, double** input, int nb_samples, double** activations);
double mse(double** y_true, const double* y_pred, int nb_samples, int output_size);
void retropropagation(const PMC* pmc, int nb_samples, double** activations, double** y_true, double** gradients_weights, double** gradients_biaises);
void update_params(const PMC* pmc, double** gradients_weights, double** gradients_biaises);

// Authentique AOP fonction de train (d'output de loss)
double train_c(PMC* pmc, double** input, double** y_true, int nb_samples);
// Usurpateur dropshippeur de double* to double** -> train_c
double train(PMC* pmc, double* input_flat, double* y_true_flat, int nb_samples, int input_size, int output_size);

double* predict(PMC* pmc, double* input_flat, int nb_samples, int input_size, int output_size);

#endif