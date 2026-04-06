#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pmc.h" 

int main(void) {
    srand(time(NULL)); 

    double inputs[4 * 2] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };

    double answers[4 * 1] = {
        0.0,
        1.0,
        1.0,
        0.0
    };
    
    double labels[4] = { 0.0, 1.0, 1.0, 0.0 };

    int sizes[] = { 2, 4, 1 };
    double learning_rate = 1.5; 
    PMC *pmc = init_pmc(sizes, 3, learning_rate);
    if (!pmc) {
        fprintf(stderr, "Erreur : init_pmc a échoué.\n");
        return 1;
    }

    int nb_epochs = 20000;

    printf("=== Entraînement XOR (%d epochs) ===\n", nb_epochs);

    for (int epoch = 0; epoch < nb_epochs; epoch++) {
        double loss = pmc_train_all(pmc, inputs, answers,
                                   4,   // nb_samples
                                   2,   // nb_features
                                   1);  // nb_outputs

        if (epoch % 1000 == 0) {
            printf("Epoch %5d | Loss = %.6f\n", epoch, loss);
        }
    }

    printf("\n=== Prédictions après entraînement ===\n");
    printf("%-12s %-12s %-12s %-12s\n",
           "x1", "x2", "Cible", "Prédiction");

    double sample_inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double targets[4] = { 0.0, 1.0, 1.0, 0.0 };

    for (int i = 0; i < 4; i++) {
        double *out = pmc_forward(pmc, sample_inputs[i]);
        printf("%-12.0f %-12.0f %-12.0f %.4f\n",
               sample_inputs[i][0],
               sample_inputs[i][1],
               targets[i],
               out[0]);
    }

    int nb_corrects = 0;
    for (int i = 0; i < 4; i++) {
        double *out = pmc_forward(pmc, sample_inputs[i]);
        int predicted = out[0] >= 0.5 ? 1 : 0;
        int actual    = (int)targets[i];
        if (predicted == actual) nb_corrects++;
    }
    printf("\nPrécision finale : %.0f%%\n", (double)nb_corrects / 4.0 * 100.0);

    free_pmc(pmc);
    return 0;
}