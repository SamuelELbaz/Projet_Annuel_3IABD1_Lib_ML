#ifndef PMC_H
#define PMC_H

#include <stddef.h>

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

double random_xavier(double low, double high);
Layer *init_layer(int nb_in, int nb_out);
void free_layer(Layer *layer);
void free_pmc(PMC *pmc);
PMC *init_pmc(int *layer_sizes, int nb_sizes, double learning_rate, int is_regression);
double func_tanh(double z);
double deriv_tanh(double z);
double sigmoid(double z);
double deriv_sigmoid(double z);
double *layer_forward(Layer *layer, double *input);
double *pmc_forward(PMC *pmc, double *input);
void delta_output(Layer *layer, double *answers);
void layer_backward(Layer *layer, Layer *next_layer);
void pmc_backward(PMC *pmc, double *answers);
void layer_update(Layer *layer, double learning_rate, int nb_samples);
void pmc_update(PMC *pmc, int nb_samples);
double pmc_train_one(PMC *pmc, double *inputs, double *answers);
double pmc_train_all(PMC *pmc, double *inputs, double *answers, int nb_samples, int nb_features, int nb_outputs);
double pmc_loss(PMC *pmc, double *inputs, double *answers, int nb_samples, int nb_features, int nb_outputs);
int pmc_predict_class(PMC *pmc, double *inputs, int nb_outputs);
double pmc_accuracy(PMC *pmc, double *inputs, double *answers_labels, int nb_samples, int nb_features, int nb_outputs);
void pmc_confusion_matrix(PMC *pmc, double *inputs, double *answers_labels, int nb_samples, int nb_features, int nb_outputs, int *confusion);
void pmc_predict_value(PMC *pmc, double *inputs, double *outputs);
int  pmc_save(PMC *pmc, const char *chemin);
PMC *pmc_load(const char *chemin);

#endif
