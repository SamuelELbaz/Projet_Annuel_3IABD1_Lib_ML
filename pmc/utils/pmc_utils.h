#ifndef PROJET_ANNUEL_3IABD_PMC_UTILS_H
#define PROJET_ANNUEL_3IABD_PMC_UTILS_H

#include "../struct/pmc_struct.h"

void free_pmc(PMC* pmc);
void free_activations(PMC* pmc, double** activations);
void free_gradients(PMC* pmc, double** gradients_weights, double** gradients_biaises);
void free_p(double* ptr);

double random_double(double a, double b);

void alloc_gradients(PMC* pmc, double*** gradients_weights, double*** gradients_biaises);

double** flat_to_double_ptr(double* flat, int nb_samples, int rows);

#endif //PROJET_ANNUEL_3IABD_PMC_UTILS_H