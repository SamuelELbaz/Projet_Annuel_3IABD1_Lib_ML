#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ===== DEBUT DES ENNUIS ===== */
/* = Struc PMC v2 = */
typedef struct {
    int nb_layers;           // Nb couches (pour malloc)
    int nb_weights;         // Nb poids (pour malloc)
    int* layers_sizes;     // Tab de taille de couches
    
    float learning_rate; // Suffisement clair 
    
    double** weights;  // Matrices de poids
    double** biases;  // Matrices de biais
} PMC;

/* ===== UTILITAIRES ===== */
/* = Liberateur Judiciaire = */
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

/* = Releveur d'erreur celeste = */
PMC* error_raiser(char* error_type, PMC* pmc, int val){
    if(val != 0) printf("Erreur : %s %d\n", error_type, val);
    else printf("Erreur : %s.\n", error_type);
    
    free_pmc(pmc);
    return NULL;
}

/* = Donneur généreur de double random = */
double random_double(double a, double b) {
    return a + (b - a) * ((double)rand() / RAND_MAX);
}

/* ===== CAUSE DE TOUTES NOS PEINES ===== */
/* = Init PMC = */
PMC* init_pmc(int* layers_sizes, int nb_layers, double learning_rate) {
    PMC* pmc = malloc(sizeof(PMC));
    if(pmc == NULL) return error_raiser("Erreur allocation PMC", pmc, 0);
    
    pmc->nb_layers     = nb_layers;
    pmc->nb_weights    = nb_layers - 1; // pas de poids sur la derniere couche (sortie)
    pmc->learning_rate = learning_rate;
    
    pmc->layers_sizes = malloc(nb_layers * sizeof(int));
    if(layers_sizes == NULL) return error_raiser("Erreur allocation layers_sizes", pmc, 0);
    for(int i = 0; i < nb_layers; i++){
        pmc->layers_sizes[i] = layers_sizes[i];
    }
    
    pmc->weights = malloc(pmc->nb_weights * sizeof(double*));
    pmc->biases  = malloc(pmc->nb_weights * sizeof(double*));
    if(pmc->weights == NULL || pmc->biases == NULL) return error_raiser("Erreur allocation weights/biases", pmc, 0);
    
    for(int i = 0; i < pmc->nb_weights; i++){
        int rows = layers_sizes[i];
        int cols = layers_sizes[i + 1];
    
        pmc->weights[i] = malloc(rows * cols * sizeof(double));
        pmc->biases[i]  = malloc(cols * sizeof(double));
        if (pmc->weights[i] == NULL || pmc->biases[i] == NULL) return error_raiser("Erreur allocation couche", pmc, i);
    
        for(int j = 0; j < rows * cols; j++){
            pmc->weights[i][j] = random_double(-0.5,0.5);
        }
        for(int j = 0; j < cols; j++){
            pmc->biases[i][j] = 0.0;
        }
    }
    
    return pmc;
}

/* = Fonction.s d'activation.s = */
double act_sigmoid(double z){
    return 1 / (1 + exp(-z));
}
double act_sigmoid_derivative(double sigmoid_res){
    return sigmoid_res * (1 - sigmoid_res);
}

/* = Erreur quadratique moyenne (MSE) = */

/* = EN AVANT = */

/* = EN ARRIERE = */

int main() {
    //init du pmc
    
    //bricoles a faire
    
    //free_pmc(pmc);
    return 0;
}
