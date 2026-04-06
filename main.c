#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pmc.h" // Assure-toi que ce fichier contient les déclarations nécessaires

// --- Déclarations (supposant que ton code est dans mlp.c / mlp.h) ---
// Si tout est dans un seul fichier, ce main peut aller à la suite directement.

int main(void) {
    srand(time(NULL)); // Graine fixe pour reproduire les résultats

    // -------------------------------------------------------
    // 1. Données XOR
    //    4 exemples, 2 features, 1 sortie
    // -------------------------------------------------------
    double inputs[4 * 2] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };

    // Sorties cibles (entre 0 et 1 pour sigmoid)
    double answers[4 * 1] = {
        0.0,
        1.0,
        1.0,
        0.0
    };
    
    // Labels entiers pour pmc_accuracy (0 ou 1)
    double labels[4] = { 0.0, 1.0, 1.0, 0.0 };

    // -------------------------------------------------------
    // 2. Création du réseau : [2, 3, 1]
    //    2 entrées → 3 neurones cachés → 1 sortie
    // -------------------------------------------------------
    int sizes[] = { 2, 4, 1 };
    double learning_rate = 1.5; // Taux d'apprentissage plus élevé pour accélérer la convergence sur XOR

    PMC *pmc = init_pmc(sizes, 3, learning_rate);
    if (!pmc) {
        fprintf(stderr, "Erreur : init_pmc a échoué.\n");
        return 1;
    }

    // -------------------------------------------------------
    // 3. Entraînement
    // -------------------------------------------------------
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

    // -------------------------------------------------------
    // 4. Évaluation finale
    // -------------------------------------------------------
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

    // Accuracy (avec seuil à 0.5 via pmc_predict_class)
    int nb_corrects = 0;
    for (int i = 0; i < 4; i++) {
        double *out = pmc_forward(pmc, sample_inputs[i]);
        int predicted = out[0] >= 0.5 ? 1 : 0;
        int actual    = (int)targets[i];
        if (predicted == actual) nb_corrects++;
    }
    printf("\nPrécision finale : %.0f%%\n", (double)nb_corrects / 4.0 * 100.0);

    // -------------------------------------------------------
    // 5. Nettoyage
    // -------------------------------------------------------
    free_pmc(pmc);
    return 0;
}