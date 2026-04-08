#include <stdio.h>
#include <stdlib.h>

#include "pmc_utils.h"

  /* ===== UTILITAIRES ===== */
 /* = Liberateur Judiciaire = */
// Pour le pmc COMPLET
void free_pmc(PMC* pmc) {
    if (pmc == NULL) return;

    for (int i = 0; i < pmc->nb_weights; i++) {
        free(pmc->weights[i]);
        free(pmc->biases[i]);
    }

    free(pmc->weights);
    free(pmc->biases);
    free(pmc->layers_sizes);
    free(pmc);
}

// pour la matrice contenant les valeurs d'activation des couches
void free_activations(PMC* pmc, double** activations) {
    for (int l = 0; l < pmc->nb_layers; l++) {
        free(activations[l]);
    }
}

// pour les matrices de gradients
void free_gradients(PMC* pmc, double** gradients_weights, double** gradients_biaises) {
    for (int i = 0; i < pmc->nb_weights; i++){
        free(gradients_weights[i]);
        free(gradients_biaises[i]);
    }
    free(gradients_weights);
    free(gradients_biaises);
}

// generaliste
void free_p(double* ptr) {
    free(ptr);
}

 /* = Donneur généreux de double random = */
double random_double(double a, double b) {
    return a + (b - a) * ((double)rand() / RAND_MAX);
}

 /* = Allocateur de Gradient intergalactic (supervisé par le grand capitaine flamme) = */
void alloc_gradients(PMC* pmc, double*** gradients_weights, double*** gradients_biaises) {
    *gradients_weights = malloc(pmc->nb_weights * sizeof(double*));
    *gradients_biaises = malloc(pmc->nb_weights * sizeof(double*));

    if (*gradients_weights == NULL || *gradients_biaises == NULL) printf("Erreur allocation matrices de gradients");

    for (int l = 0; l < pmc->nb_weights; l++) {
        int rows = pmc->layers_sizes[l];
        int cols = pmc->layers_sizes[l + 1];

        (*gradients_weights)[l] = malloc(rows * cols * sizeof(double));
        (*gradients_biaises)[l] = malloc(cols * sizeof(double));

        if ((*gradients_weights)[l] == NULL || (*gradients_biaises)[l] == NULL) printf("Erreur allocation matrices de gradients couche");
    }
}

 /* = Convertisseur de * to ** (pour input python) = */
// Deplatisseur de matrice
double** flat_to_double_ptr(double* flat, int nb_samples, int rows) {
    double** res = malloc(nb_samples * sizeof(double*));
    if (res == NULL) {
        printf("Erreur allocation flat_to_double_ptr\n");
        return NULL;
    }

    for (int i = 0; i < nb_samples; i++) {
        res[i] = &flat[i * rows];
    }

    return res;
}