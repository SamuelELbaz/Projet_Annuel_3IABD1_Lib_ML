#ifndef PMC_STRUCT_H
#define PMC_STRUCT_H

/* ===== DEBUT DES ENNUIS ===== */
/* = Struc PMC v2 = */
typedef struct {
    int nb_layers;           // Nb couches (pour malloc)
    int nb_weights;         // Nb poids (pour malloc)
    int* layers_sizes;     // Tab de taille de couches

    double learning_rate; // Suffisement clair

    double** weights;  // Matrices de poids
    double** biases;  // Matrices de biais
} PMC;

#endif