#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define OUTPUT_SIGMOID 0
#define OUTPUT_TANH 1
#define OUTPUT_LINEAR 2

typedef struct {
    int nb_in; //Nombre d'entrées reçues pas la couche
    int nb_out; //Nombre de neurones de la couche
    int output_type; //Type de la fonction d'activation de la couche de sortie

    double *weights; //Poids de la couche
    double *biases; //Biais de la couche

    double *activations; //Valeurs après activation
    double *entries; //Pointeurs vers les entrées de la couche

    double *d_weights; //Gradients des poids
    double *d_biases; //Gradients des biais
    double *delta; //Signal d'erreur de la couche

}Layer;

typedef struct {
    Layer **layers; //Tableau de pointeurs vers les couches
    int nb_layers; //Nombre de couches
    double learning_rate; //Taux d'apprentissage
} PMC;

double random_xavier(double low, double high) {
    return low + ((double)rand() / (double)RAND_MAX) * (high - low);
}

Layer *init_layer(int nb_in, int nb_out) {
    Layer *layer = malloc(sizeof(Layer));
    if(!layer){
        return NULL;
    }

    layer->nb_in = nb_in;
    layer->nb_out = nb_out;
    layer->output_type = OUTPUT_TANH;

    layer->weights = malloc(nb_in * nb_out * sizeof(double));
    if(!layer->weights){
        free(layer);
        return NULL;
    }

    layer->biases = calloc(nb_out, sizeof(double));
    if(!layer->biases){
        free(layer->weights);
        free(layer);
        return NULL;
    }

    layer->activations = calloc(nb_out, sizeof(double));
    if(!layer->activations){
        free(layer->biases);
        free(layer->weights);
        free(layer);
        return NULL;
    }

    layer->entries = NULL; //Pointe vers les entrées reçues pendant la propagation

    layer->d_weights = calloc(nb_in * nb_out, sizeof(double));
    if(!layer->d_weights){
        free(layer->activations);
        free(layer->biases);
        free(layer->weights);
        free(layer);
        return NULL;
    }

    layer->d_biases = calloc(nb_out, sizeof(double));
    if(!layer->d_biases){
        free(layer->d_weights);
        free(layer->activations);
        free(layer->biases);
        free(layer->weights);
        free(layer);
        return NULL;
    }

    layer->delta = calloc(nb_out, sizeof(double));
    if(!layer->delta){
        free(layer->d_biases);
        free(layer->d_weights);
        free(layer->activations);
        free(layer->biases);
        free(layer->weights);
        free(layer);
        return NULL;
    }

    //Initialisation des poids avec la méthode de Xavier

    double limit = sqrt(6.0 / (nb_in + nb_out));
    for (int i = 0; i < nb_in * nb_out; i++) {
        layer->weights[i] = random_xavier(-limit, limit);
    }

    return layer;
}

void free_layer(Layer *layer) {
    if (layer) {
        free(layer->weights);
        free(layer->biases);
        free(layer->activations);
        free(layer->d_weights);
        free(layer->d_biases);
        free(layer->delta);
        free(layer);
    }
}

void free_pmc(PMC *pmc) {
    if (pmc) {
        for (int i = 0; i < pmc->nb_layers; i++) {
            free_layer(pmc->layers[i]);
        }
        free(pmc->layers);
        free(pmc);
    }
}


PMC *init_pmc(int *layer_sizes, int nb_sizes, double learning_rate, int is_regression) {

    PMC *pmc = malloc(sizeof(PMC));
    if(!pmc){
        return NULL;
    }

    pmc->nb_layers = nb_sizes - 1; 
    pmc->learning_rate = learning_rate;

    pmc->layers = malloc(pmc->nb_layers * sizeof(Layer*));
    if(!pmc->layers){
        free(pmc);
        return NULL;
    }

    for(int i = 0; i < pmc->nb_layers; i++){
        pmc->layers[i] = NULL;
    }

    for(int i = 0; i < pmc->nb_layers; i++){
        pmc->layers[i] = init_layer(layer_sizes[i], layer_sizes[i+1]);
        if(!pmc->layers[i]){
            free_pmc(pmc);
            return NULL;
        }
    }

    pmc->layers[pmc->nb_layers - 1]->output_type = is_regression ? OUTPUT_LINEAR : OUTPUT_TANH;

    return pmc;
}

double func_tanh(double z) {
    return tanh(z);
}

double deriv_tanh(double z) {
    return 1.0 - z * z;
}

double sigmoid(double z){
    return 1.0 / (1.0 + exp(-z));
}

double deriv_sigmoid(double z){
    return z * (1.0 - z);
}

double *layer_forward(Layer *layer, double *input){
    layer->entries = input;

    for(int i = 0; i < layer->nb_out; i++){
        double activation = layer->biases[i];

        for(int j = 0; j < layer->nb_in; j++){
            activation += layer->weights[i * layer->nb_in + j] * input[j];
        }
        if(layer->output_type == OUTPUT_SIGMOID){
            layer->activations[i] = sigmoid(activation);
        } else if(layer->output_type == OUTPUT_TANH){
            layer->activations[i] = func_tanh(activation);
        } else { //OUTPUT_LINEAR
            layer->activations[i] = activation;
        }
    }

    return layer->activations;
}

double *pmc_forward(PMC *pmc, double *input){
    double *output = input;

    for(int i = 0; i < pmc->nb_layers; i++){
        output = layer_forward(pmc->layers[i], output);
    }

    return output;
}


void delta_output(Layer *layer, double *answers){
    for(int i = 0; i < layer->nb_out; i++){
        double activation = layer->activations[i];
        double error = activation - answers[i];
        double deriv;
        if(layer->output_type == OUTPUT_SIGMOID){
            deriv = deriv_sigmoid(activation);
        } else if(layer->output_type == OUTPUT_TANH){
            deriv = deriv_tanh(activation);
        } else { //OUTPUT_LINEAR
            deriv = 1.0;
        }

        layer->delta[i] = error * deriv;
    }
}

void layer_backward(Layer *layer, Layer *next_layer){

    // Delta de la couche courante
    for(int i = 0; i < layer->nb_out; i++){
        double signal = 0.0;

        for(int j = 0; j < next_layer->nb_out; j++){
            signal += next_layer->weights[j * next_layer->nb_in + i] * next_layer->delta[j];
        }
        double deriv;
        if(layer->output_type == OUTPUT_SIGMOID){
            deriv = deriv_sigmoid(layer->activations[i]);
        } else if(layer->output_type == OUTPUT_TANH){
            deriv = deriv_tanh(layer->activations[i]);
        } else { //OUTPUT_LINEAR
            deriv = 1.0;
        }
        layer->delta[i] = signal * deriv;
    }

    // Accumulation des gradients d_weights

    for(int i = 0; i < layer->nb_out; i++){
        for(int j = 0; j < layer->nb_in; j++){
            layer->d_weights[i * layer->nb_in + j] += layer->delta[i] * layer->entries[j];
        }
    }

    // Accumulation des gradients d_biases  
    
    for(int i = 0; i < layer->nb_out; i++){
        layer->d_biases[i] += layer->delta[i];
    }
}

void pmc_backward(PMC *pmc, double *answers) {
    // Delta de la couche de sortie
    delta_output(pmc->layers[pmc->nb_layers - 1], answers);

    // Accumulation des gradients de la couche de sortie
    Layer *out_layer = pmc->layers[pmc->nb_layers - 1];
    for (int i = 0; i < out_layer->nb_out; i++) {
        for (int j = 0; j < out_layer->nb_in; j++) {
            out_layer->d_weights[i * out_layer->nb_in + j] +=
                out_layer->delta[i] * out_layer->entries[j];
        }
        out_layer->d_biases[i] += out_layer->delta[i];
    }

    // Deltas et gradients des couches cachées
    for (int i = pmc->nb_layers - 2; i >= 0; i--) {
        layer_backward(pmc->layers[i], pmc->layers[i + 1]);
    }
}

void layer_update(Layer *layer, double learning_rate, int nb_samples){
    double scale = learning_rate / (double)nb_samples;

    for(int i = 0; i < layer->nb_out * layer->nb_in; i++){
        layer->weights[i] -= scale * layer->d_weights[i];
        layer->d_weights[i] = 0.0; //Reset du gradient
    }

    for(int i = 0; i < layer->nb_out; i++){
        layer->biases[i] -= scale * layer->d_biases[i];
        layer->d_biases[i] = 0.0; //Reset du gradient
    }
}

void pmc_update(PMC *pmc, int nb_samples){
    for(int i = 0; i < pmc->nb_layers; i++){
        layer_update(pmc->layers[i], pmc->learning_rate, nb_samples);
    }
}

double pmc_train_one(PMC *pmc, double *inputs, double *answers){
    double *output = pmc_forward(pmc, inputs);
    Layer *output_layer = pmc->layers[pmc->nb_layers -1];
    double loss = 0.0;

    for(int i = 0; i < output_layer->nb_out; i++){
        double error = output[i] - answers[i];
        loss += 0.5 * error * error;
    }

    pmc_backward(pmc, answers);

    pmc_update(pmc, 1); 

    return loss;
}

double pmc_train_all(PMC *pmc, double *inputs, double *answers,
                     int nb_samples, int nb_features, int nb_outputs) {
    double total_loss = 0.0;

    for (int i = 0; i < nb_samples; i++) {
        double *input  = inputs  + i * nb_features;
        double *answer = answers + i * nb_outputs;

        double *output = pmc_forward(pmc, input);

        for (int j = 0; j < nb_outputs; j++) {
            double error = output[j] - answer[j];
            total_loss += 0.5 * error * error;
        }

        pmc_backward(pmc, answer); 
    }

    pmc_update(pmc, nb_samples);

    return total_loss / (double)nb_samples;
}

double pmc_loss(PMC *pmc, double *inputs, double *answers, int nb_samples, int nb_features, int nb_outputs){
    double total_loss = 0.0;

    for(int i = 0; i < nb_samples; i++){
        double *input = inputs + i * nb_features;
        double *answer = answers + i * nb_outputs;

        double *output = pmc_forward(pmc, input);

        for(int j = 0; j < nb_outputs; j++){
            double error = output[j] - answer[j];
            total_loss += 0.5 * error * error;
        }
    }

    return total_loss / (double)nb_samples;
}

int pmc_predict_class(PMC *pmc, double *inputs, int nb_outputs){
    double *output = pmc_forward(pmc, inputs);

    if(nb_outputs == 1) {
        return output[0] >= 0.0 ? 0 : 1;
    }

    int max_index = 0;
    double max_score = output[0];

    for(int i = 1; i < nb_outputs; i++){
        if(output[i] > max_score){
            max_score = output[i];
            max_index = i;
        }
    }

    return max_index;
}

double pmc_accuracy(PMC *pmc, double *inputs, double *answers_labels, int nb_samples, int nb_features, int nb_outputs){
    int nb_corrects = 0;
    
    for(int i = 0; i < nb_samples; i++){
        double *input = inputs + i * nb_features;
        int predicted = pmc_predict_class(pmc, input, nb_outputs);
        int actual_label = answers_labels[i];

        if(predicted == actual_label){
            nb_corrects ++;
        }
    }

    return (double)nb_corrects / (double)nb_samples;
}

void pmc_confusion_matrix(PMC *pmc, double *inputs, double *answers_labels, int nb_samples, int nb_features, int nb_outputs, int *confusion){
    int nb_classes = (nb_outputs == 1) ? 2 : nb_outputs;

    for(int i = 0; i < nb_samples; i++){
        double *input = inputs + i * nb_features;
        int prediction = pmc_predict_class(pmc, input, nb_outputs);
        int actual_label = answers_labels[i];

        confusion[actual_label * nb_classes + prediction]++;
    }
}

void pmc_predict_value(PMC *pmc, double *inputs, double *outputs){
    double *output = pmc_forward(pmc, inputs);
    Layer *last = pmc->layers[pmc->nb_layers - 1];
    for(int i = 0; i < last->nb_out; i++){
        outputs[i] = output[i];
    }
}


int pmc_save(PMC *pmc, const char *chemin) {
    FILE *f = fopen(chemin, "w");
    if (!f) {
        fprintf(stderr, "pmc_save: impossible d'ouvrir %s\n", chemin);
        return -1;
    }

    /*En-tete */
    fprintf(f, "# PMC - Modele sauvegarde\n");
    fprintf(f, "n_layers %d\n", pmc->nb_layers);
    fprintf(f, "lr %.10f\n", pmc->learning_rate);

    /* Tailles des couches */
    fprintf(f, "sizes");
    fprintf(f, " %d", pmc->layers[0]->nb_in);   /* entree */
    for (int l = 0; l < pmc->nb_layers; l++)
        fprintf(f, " %d", pmc->layers[l]->nb_out);
    fprintf(f, "\n");

    /* Poids et biais de chaque couche */
    for (int l = 0; l < pmc->nb_layers; l++) {
        Layer *layer = pmc->layers[l];

        fprintf(f, "# Couche %d : %d -> %d\n",
                l, layer->nb_in, layer->nb_out);

        /* Poids : n_out * n_in valeurs */
        fprintf(f, "weights");
        for (int k = 0; k < layer->nb_out * layer->nb_in; k++)
            fprintf(f, " %.10f", layer->weights[k]);
        fprintf(f, "\n");

        /* Biais : n_out valeurs */
        fprintf(f, "biases");
        for (int i = 0; i < layer->nb_out; i++)
            fprintf(f, " %.10f", layer->biases[i]);
        fprintf(f, "\n");
    }

    fclose(f);
    fprintf(stdout, "Modele sauvegarde dans : %s\n", chemin);
    return 0;
}


PMC *pmc_load(const char *chemin) {
    FILE *f = fopen(chemin, "r");
    if (!f) {
        fprintf(stderr, "pmc_load: fichier introuvable : %s\n", chemin);
        return NULL;
    }

    int    n_layers = 0;
    double lr       = 0.0;
    int    sizes[32];   /* max 32 couches */
    char   line[64];

    /* Lecture de l'en-tete */
    while (fscanf(f, "%s", line) == 1) {

        if (line[0] == '#') {
            /* Ligne de commentaire - ignorer jusqu'a la fin de ligne */
            char buf[1024];
            fgets(buf, sizeof(buf), f);
            continue;
        }

        if (strcmp(line, "n_layers") == 0) {
            fscanf(f, "%d", &n_layers);

        } else if (strcmp(line, "lr") == 0) {
            fscanf(f, "%lf", &lr);

        } else if (strcmp(line, "sizes") == 0) {
            /* n_layers + 1 tailles (entree incluse) */
            for (int i = 0; i <= n_layers; i++)
                fscanf(f, "%d", &sizes[i]);

        } else if (strcmp(line, "weights") == 0) {
            /* Determine quelle couche on est en train de lire */
            /* On se base sur le compteur de couches lues      */
            /* -> gere dans la boucle ci-dessous               */
            break;
        }
    }

    /* Creation du modele avec les bonnes dimensions */
    PMC *pmc = init_pmc(sizes, n_layers + 1, lr, 0);
    if (!pmc) {
        fclose(f);
        return NULL;
    }

    /* Rechargement des poids couche par couche */
    /* On repart du debut du fichier            */
    rewind(f);

    int couche_courante = -1;

    while (fscanf(f, "%s", line) == 1) {

        if (line[0] == '#') {
            char buf[1024];
            fgets(buf, sizeof(buf), f);
            /* Detecter "# Couche X" pour suivre la couche courante */
            if (strncmp(buf, " Couche", 7) == 0)
                couche_courante++;
            continue;
        }

        if (strcmp(line, "weights") == 0 && couche_courante >= 0) {
            Layer *layer = pmc->layers[couche_courante];
            for (int k = 0; k < layer->nb_out * layer->nb_in; k++)
                fscanf(f, "%lf", &layer->weights[k]);

        } else if (strcmp(line, "biases") == 0 && couche_courante >= 0) {
            Layer *layer = pmc->layers[couche_courante];
            for (int i = 0; i < layer->nb_out; i++)
                fscanf(f, "%lf", &layer->biases[i]);
        }
    }

    fclose(f);
    fprintf(stdout, "Modele charge depuis : %s\n", chemin);
    return pmc;
}
