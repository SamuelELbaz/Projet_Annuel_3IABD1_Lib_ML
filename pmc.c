#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int nb_in; //Nombre d'entrées reçues pas la couche
    int nb_out; //Nombre de neurones de la couche

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


PMC *init_pmc(int *layer_sizes, int nb_sizes, double learning_rate) {

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

    return pmc;
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

        layer->activations[i] = sigmoid(activation);
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
        double deriv_sig = deriv_sigmoid(activation);

        layer->delta[i] = error * deriv_sig;
    }
}

void layer_backward(Layer *layer, Layer *next_layer){

    // Delta de la couche courante
    for(int i = 0; i < layer->nb_out; i++){
        double signal = 0.0;

        for(int j = 0; j < next_layer->nb_out; j++){
            signal += next_layer->weights[j * next_layer->nb_in + i] * next_layer->delta[j];
        }

        layer->delta[i] = signal * deriv_sigmoid(layer->activations[i]);
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

    for(int i = 0; i < nb_samples; i++){
        double *input = inputs + i * nb_features;
        int prediction = pmc_predict_class(pmc, input, nb_outputs);
        int actual_label = answers_labels[i];

        confusion[actual_label * nb_outputs + prediction]++;
    }
}