/*
 * Modèle Linéaire de Classification
 */

#include <stdio.h>
#include <stdlib.h>

#define MAX_SAMPLES 100
#define MAX_FEATURES 5

double X[MAX_SAMPLES][MAX_FEATURES];
int Y[MAX_SAMPLES];

double W[MAX_FEATURES];
double bias = 0.0;
double lr = 0.01;

int N = 0;
int D = 0;

double predict(double *x)
{
    double y = bias;
    for(int i = 0; i < D; i++)
        y += W[i] * x[i];
    return y;
}

int classify(double *x)
{
    return predict(x) >= 0.5 ? 1 : 0;
}

void train(int epochs)
{
    for(int e = 0; e < epochs; e++) {
        double loss = 0.0;
        for(int i = 0; i < N; i++) {
            int pred = classify(X[i]);
            int err = Y[i] - pred;
            loss += err * err;

            for(int j = 0; j < D; j++)
                W[j] += lr * err * X[i][j];

            bias += lr * err;
        }
        loss /= N;
        if(e % 10 == 0 || e == epochs - 1)
            printf("Epoch %d/%d | Loss: %.6f\n", e, epochs - 1, loss);
    }
}

double accuracy()
{
    int correct = 0;
    for(int i = 0; i < N; i++)
        if(classify(X[i]) == Y[i])
            correct++;

    return (double)correct / N;
}

void test_lineaire()
{
    printf("\n=== Test dataset lineaire ===\n");

    D = 2;
    N = 6;

    double tmp[6][2] = {
        {2, 2}, {3, 3}, {4, 2},
        {-2, -2}, {-3, -3}, {-4, -2}
    };

    int lab[6] = {1, 1, 1, 0, 0, 0};

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < D; j++)
            X[i][j] = tmp[i][j];
        Y[i] = lab[i];
    }

    for(int i = 0; i < D; i++) W[i] = 0.0;
    bias = 0.0;

    train(50);
    printf("Accuracy: %.2f%%\n", accuracy() * 100);
    printf("Poids: W0=%.4f, W1=%.4f | Bias=%.4f\n", W[0], W[1], bias);
}

void test_KO()
{
    printf("\n=== Test dataset KO (non lineaire) ===\n");

    D = 2;
    N = 5;

    double tmp[5][2] = {
        {0, 0},
        {1, 0},
        {0, 1},
        {-1, 0},
        {0, -1}
    };

    int lab[5] = {0, 1, 1, 1, 1};

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < D; j++)
            X[i][j] = tmp[i][j];
        Y[i] = lab[i];
    }

    for(int i = 0; i < D; i++) W[i] = 0.0;
    bias = 0.0;

    train(50);
    printf("Accuracy: %.2f%%\n", accuracy() * 100);
    printf("Poids: W0=%.4f, W1=%.4f | Bias=%.4f\n", W[0], W[1], bias);
}

void test_transformation()
{
    printf("\n=== Transformation non lineaire ===\n");

    D = 3;
    N = 5;

    double base[5][2] = {
        {0, 0},
        {1, 0},
        {0, 1},
        {-1, 0},
        {0, -1}
    };

    int lab[5] = {0, 1, 1, 1, 1};

    for(int i = 0; i < N; i++) {
        double x1 = base[i][0];
        double x2 = base[i][1];

        X[i][0] = x1;
        X[i][1] = x2;
        X[i][2] = x1*x1 + x2*x2;

        Y[i] = lab[i];
    }

    for(int i = 0; i < D; i++) W[i] = 0.0;
    bias = 0.0;

    train(50);
    printf("Accuracy: %.2f%%\n", accuracy() * 100);
    printf("Poids: W0=%.4f, W1=%.4f, W2=%.4f | Bias=%.4f\n", W[0], W[1], W[2], bias);
}

int main()
{
    printf("\n=== Modele lineaire ===\n");

    test_lineaire();
    test_KO();
    test_transformation();

    return 0;
}