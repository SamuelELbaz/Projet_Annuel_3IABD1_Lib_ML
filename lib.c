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
void free_gradients(PMC* pmc, double** gradients_weights, double** gradients_biaises){
    for (int i = 0; i < pmc->nb_weights; i++){
        free(gradients_weights[i]);
        free(gradients_biaises[i]);
    }
    free(gradients_weights);
    free(gradients_biaises);
}

 /* = Releveur d'erreur celeste = */
PMC* error_raiser(char* error_type, PMC* pmc, int val){
    if(val != 0) printf("Erreur : %s %d\n", error_type, val);
    else printf("Erreur : %s.\n", error_type);
    
    free_pmc(pmc);
    return NULL;
}

 /* = Donneur généreux de double random = */
double random_double(double a, double b){
    return a + (b - a) * ((double)rand() / RAND_MAX);
}

 /* = Allocateur de Gradient intergalactic (supervisé par le grand capitaine flamme) = */
PMC* alloc_gradients(PMC* pmc, double*** gradients_weights, double*** gradients_biaises) {
    *gradients_weights = malloc(pmc->nb_weights * sizeof(double*));
    *gradients_biaises = malloc(pmc->nb_weights * sizeof(double*));

    if (*gradients_weights == NULL || *gradients_biaises == NULL) return error_raiser("Erreur allocation matrices de gradients", pmc, 0);

    for (int l = 0; l < pmc->nb_weights; l++) {
        int rows = pmc->layers_sizes[l];
        int cols = pmc->layers_sizes[l + 1];

        (*gradients_weights)[l] = malloc(rows * cols * sizeof(double));
        (*gradients_biaises)[l] = malloc(cols * sizeof(double));

        if ((*gradients_weights)[l] == NULL || (*gradients_biaises)[l] == NULL) return error_raiser("Erreur allocation matrices de gradients couche", pmc, l);
    }
}

  /* ===== CAUSE DE TOUTES NOS PEINES ===== */
 /* = Init PMC = */
// layers_sizes -> [2, 2, 1] | nb_layers -> 3 ([0, 1, 2]) | learning_rate -> Suffisement clair mdr
PMC* init_pmc(int* layers_sizes, int nb_layers, double learning_rate){
    PMC* pmc = malloc(sizeof(PMC));
    if (pmc == NULL) return error_raiser("Erreur allocation PMC", pmc, 0);
    
    pmc->nb_layers     = nb_layers;
    pmc->nb_weights    = nb_layers - 1; // pas de poids sur la derniere couche (sortie) (d'ou le -1)
    pmc->learning_rate = learning_rate;
    
    pmc->layers_sizes = malloc(nb_layers * sizeof(int));
    if (layers_sizes == NULL) return error_raiser("Erreur allocation layers_sizes", pmc, 0);
    for (int i = 0; i < nb_layers; i++){
        pmc->layers_sizes[i] = layers_sizes[i];
    }
    
    pmc->weights = malloc(pmc->nb_weights * sizeof(double*));
    pmc->biases  = malloc(pmc->nb_weights * sizeof(double*));
    if (pmc->weights == NULL || pmc->biases == NULL) return error_raiser("Erreur allocation weights/biases", pmc, 0);
    
    for (int i = 0; i < pmc->nb_weights; i++){
        int rows = layers_sizes[i];
        int cols = layers_sizes[i + 1];
    
        pmc->weights[i] = malloc(rows * cols * sizeof(double));
        pmc->biases[i]  = malloc(cols * sizeof(double));
        if (pmc->weights[i] == NULL || pmc->biases[i] == NULL) return error_raiser("Erreur allocation couche", pmc, i);
    
        for (int j = 0; j < rows * cols; j++){
            pmc->weights[i][j] = random_double(-0.1,0.1);
        }
        for (int j = 0; j < cols; j++){
            pmc->biases[i][j] = 0.0;
        }
    }
    
    return pmc;
}

 /* = Fonction.s d'activation.s = */
// Sigmoid
double act_sigmoid(double z){
    return 1 / (1 + exp(-z));
}
// Sigmoid' -> prend en entrée le resultat de act_sigmoid(double z)
double act_sigmoid_derivative(double sigmoid_res){
    return sigmoid_res * (1 - sigmoid_res);
}

 /* = EN AVANT = */
// Batch
PMC* propagation(PMC* pmc, double** input, int nb_samples, double** activations){
    for (int l = 0; l < pmc->nb_layers; l++){
        activations[l] = malloc(nb_samples * pmc->layers_sizes[l] * sizeof(double));
        if (activations[l] == NULL) return error_raiser("Erreur allocation activations couche", pmc, l);
    }
    
     // activations[0] (premiere couche) = layers_sizes[0] (couche d'entrée) -> on copie
    // avant de commencer a remplir la matrice d'activation
    for (int s = 0; s < nb_samples; s++){
        for (int i =0; i < pmc->layers_sizes[0]; i++){
            activations[0][s * pmc->layers_sizes[0] + i] = input[s][i];
        }
    }
    
    for (int l = 0; l < pmc->nb_weights; l++) {
         // sombre et ingenieuse technique pour naviguer
        // dans une matrice 2d applatie en 1d
        int rows = pmc->layers_sizes[l];
        int cols = pmc->layers_sizes[l + 1];
        
        // pour tout les samples d'entrée (passé en activations[0][])
        for (int s = 0; s < nb_samples; s++){
            for (int j = 0; j < cols; j++) {
                // combinaison lineaire + biais (va ensuite passer dans fn act)
                double sum = 0.0;
    
                for (int k = 0; k < rows; k++) {
                    double act = activations[l][s * rows + k];
                    double wei = pmc->weights[l][k * cols + j];
                    
                    sum = act * wei;
                }
                // ajout du biais
                sum += pmc->biases[l][j];
                // activation de cette combinaison lineaire + biais
                activations[l + 1][s * cols + j] = act_sigmoid(sum);
            }
        }
    }
}

 /* = Erreur quadratique moyenne (MSE) = */
// Batch
double mse(double **y_true, double **y_pred, int samples, int output_size) {
    double total = 0.0;

    for (int s = 0; s < samples; s++) {
        for (int i = 0; i < output_size; i++) {
            double error = y_true[s][i] - y_pred[s][i];
            total += error * error;
        }
    }

    return total / (samples * output_size);
}

 /* = EN ARRIERE = */
// Batch 
void retropropagation(PMC* pmc, double** input, int nb_samples, double** activations, double** y_true, double** gradients_weights, double** gradients_biaises){
    
}

int main() {
    //init du pmc
    int ls[] = {2,2,1};
    PMC* pmc =init_pmc(ls,3,0.5);
    printf("%d",pmc->nb_weights);
    
    //bricoles a faire
    
    free_pmc(pmc);
    return 0;
}
