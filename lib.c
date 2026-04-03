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
double act_sigmoid(double a){
    return 1 / (1 + exp(-a));
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
                    
                    // 
                    sum += act * wei;
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
// Beaucoup de commentaires pour eviter de perdre les connaissances duement accumulés a la suite d'heures d'incomprehensions
void retropropagation(PMC* pmc, double** input, int nb_samples, double** activations, double** y_true, double** gradients_weights, double** gradients_biaises){
    
    int idx_output_layer  = pmc->nb_layers - 1;
    int output_layer_size = pmc->layers_sizes[idx_output_layer];
    
    // delta = (valeur_activation_sortie - y_true(valeur visé)) * act_sigmoid_derivative(activation_sortie);
    double* delta = malloc(nb_samples * output_layer_size * sizeof(double));
    if (delta == NULL) return print("Erreur allocation delta");
    
    for (int s = 0; s < nb_samples; s++){
        // POUR TOUT LES SAMPLES    
        for (int i = 0; i < output_layer_size; i++){
            // Pour chacun des neurones de la couches
            // On recup l'Activation du neurone i pour le sample s
            double val_acti_sortie = activations[idx_output_layer][s * output_layer_size + i];
            // On recup la valeur attendue pour ce sample et ce neurone de sortie
            double val_true = y_true[s][i];
            
             // ET LA on remplie notre matrice de delta
            //delta = (prediction - cible) * sigmoid'(sigmoid(a))
            delta[s * output_layer_size + i] = (val_acti_sortie - val_true) * act_sigmoid_derivative(val_acti_sortie);
        }
        
    }
    
    
    int idx_last_weight_matrix = pmc->nb_weights - 1;
      // Rappel layers_sizes => [2,2,1] -> (nb neurones par couche)
     // Determiner la taille de la last_weight_matrix (2x1) 
    // On aurait ici idx = 1 | prev == 2 (layers_sizes[1] = 2) | curr == 2 (layers_sizes[2] == 1)
    int prev_layer_size = pmc->layers_sizes[idx_last_weight_matrix];
    int curr_layer_size = pmc->layers_sizes[idx_last_weight_matrix + 1];
    
    for (int i = 0; i < prev_layer_size; i++){
        // Pour tout les neurones de la couche precedente
        for (int j = 0; j < curr_layer_size; j++{
            // Pour tout les neurones de la couche actuelle (sortie)
            
            double gradient_sum = 0.0;
            
            // On additionne 
            for (int s = 0; s < nb_samples; s++) {
                //Pour chaque sample
                
                // Activation du neurone i dans la couche precedente pour le sample s
                double val_acti_prev = activations[idx_last_weight_matrix][s * prev_layer_size + i];
                
                // Delta du neurone j de sortie pour le sample s
                double val_delta = delta[s * curr_layer_size + j];
                
                // Calcule du gradient de cette relation i j pour le sample s
                gradient_sum += val_acti_prev * val_delta;
            }
            
            // Gradient moyen sur le batch parce qu'on batch
            gradients_weights[idx_last_weight_matrix][i * curr_layer_size + j] = gradient_sum / nb_samples;
        }
    }
    
    // gradients_biaises d'un neurone j = moyenne sur les samples de delta(sample, j)
    for (int j = 0; j < curr_layer_size; j++) {
        // Pour chaque neurone de la couche actuelle (sortie)
        
        double delta_sum = 0.0;
        
        // On additionne les deltas 
        for (int s = 0; s < nb_samples; s++) {
            // Pour tous les samples
            delta_sum += delta[s * curr_layer_size + j];
        }
        
        // Gradient moyen du biais sur le batch
        gradients_biaises[idx_last_weight_matrix][j] = delta_sum / nb_samples;
    }
    
    
    // nb_weights - 2 pour partir de l'avant dernier matrice de poid -> remonter jusqu'a weights[0]
    for (int idx_weight_matrix = pmc->nb_weights - 2; idx_weight_matrix >= 0; idx_weight_matrix--) {
        // On passe donc sur chacune de ces gourmandes matrices de poids
        
          // Taille de la couche dont on veut le delta
         // on aurait (pour [2,2,1]) idx_weight_matrix = 0 (car = nb_weights - 2 (et nb_weights = nb_layers - 1 (avec nb_layers = 3)))
        // donc hidden_layer_size = 2 -> layers_sizes[1] = 2 -> on est bien (a la premiere iteration) sur la derniere couche caché
        int hidden_layer_size = pmc->layers_sizes[idx_weight_matrix + 1];
          
         // Taille de la couche dont on a deja le delta
        // next_layer_size = 1 -> layers_sizes[0 + 2] = 1
        int next_layer_size = pmc->layers_sizes[idx_weight_matrix + 2];
        
        // Nouveau tableau de delta pour la couche cachée courante
        double* new_delta = malloc(nb_samples * hidden_layer_size * sizeof(double));
        if (new_delta == NULL) { // pas d'error_raiser, ca free tout le pmc mdr, a revoir
            printf("Erreur allocation new_delta\n");
            free(delta);
            return;
        }
        
        for (int s = 0; s < nb_samples; s++) {
            // Pour chaque sample

            for (int i = 0; i < hidden_layer_size; i++) {
                // Pour chaque neurone de la couche cachee
                
                double weighted_delta_sum = 0.0;
                
                  // On additionne les deltas de la couche suivante
                 // Ponderes par les poids reliant le neurone cache courant aux neurones de la couche suivante
                // somme += delta_next[j] * weight[i,j]
                for (int j = 0; j < next_layer_size; j++) {
                    
                    double val_delta_next = delta[s * next_layer_size + j];
                    
                    double val_weight_to_next = pmc->weights[idx_weight_matrix + 1][i * next_layer_size + j];
                    
                    weighted_delta_sum += val_delta_next * val_weight_to_next;
                }
                
                // On recup la valeur d'activation du neurone cache courant
                double val_acti_hidden = activations[idx_weight_matrix + 1][s * hidden_layer_size + i];
                
                // Delta du neurone cache = matrice des delta ponderee * sigmoid'(valeur d'activation du neuronne cache courant)
                new_delta[s * hidden_layer_size + i] = weighted_delta_sum * act_sigmoid_derivative(val_acti_hidden);
            }
        }
        
        free(delta);
        delta = new_delta;
        
        
           // Gradients des poids de cette couche
          // Pour [2,2,1] avec idx_weight_matrix = 0 :
         // prev_layer_size_local = layers_sizes[0] = 2
        //  curr_layer_size_local = layers_sizes[1] = 2
        int prev_layer_size_local = pmc->layers_sizes[idx_weight_matrix];
        int curr_layer_size_local = pmc->layers_sizes[idx_weight_matrix + 1];
        
        for (int i = 0; i < prev_layer_size_local; i++) {
            // Pour chaque neurone de la couche précédente
            for (int j = 0; j < curr_layer_size_local; j++) {
                // Pour chaque neurone de la couche courante
                
                double gradient_sum = 0.0;
                
                for (int s = 0; s < nb_samples; s++) {
                    // On additionne la contribution de chaque sample
                    
                    double val_acti_prev = activations[idx_weight_matrix][s * prev_layer_size_local + i];
                    
                    double val_delta_curr = delta[s * curr_layer_size_local + j];
                    
                    gradient_sum += val_acti_prev * val_delta_curr;
                }
                
                // Gradient moyen sur le batch
                gradients_weights[idx_weight_matrix][i * curr_layer_size_local + j] = gradient_sum / nb_samples;
            }
        }
        
         // Gradients des biais de cette couche
        
        for (int j = 0; j < curr_layer_size_local; j++) 
            // Pour chaque neurone de la couche courante{
            
            double delta_sum = 0.0;
            
            for (int s = 0; s < nb_samples; s++) {
                // On additionne les deltas de tous les samples
                delta_sum += delta[s * curr_layer_size_local + j];
            }
            
            // Gradient moyen du biais sur le batch
            gradients_biaises[idx_weight_matrix][j] = delta_sum / nb_samples;
        }
    }
}

/* = MISE A JOUR ECCLESIASTIQUE DES PARAMETRES (fois le learning rate) = */
void update_params(PMC* pmc, double** gradients_weights, double** gradients_biaises) {
    for (int l = 0; l < pmc->nb_weights; l++) {
        int rows = pmc->layers_sizes[l];
        int cols = pmc->layers_sizes[l + 1];

        // Mise à jour des poids
        for (int i = 0; i < rows * cols; i++) {
            pmc->weights[l][i] -= pmc->learning_rate * gradients_weights[l][i];
        }

        // Mise à jour des biais
        for (int j = 0; j < cols; j++) {
            pmc->biases[l][j] -= pmc->learning_rate * gradients_biaises[l][j];
        }
    }
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












