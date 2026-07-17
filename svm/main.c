#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "svm.h"

double** alloc_matrix(
    int n,
    int d)
{
    double **M =
        malloc(n * sizeof(double*));

    for(int i = 0; i < n; i++)
    {
        M[i] = malloc(d * sizeof(double));
    }

    return M;
}

void free_matrix(
    double **M,
    int n)
{
    for(int i = 0; i < n; i++)
    {
        free(M[i]);
    }
    free(M);
}

double randu(void)
{
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

double pymod(
    double a,
    double b)
{
    double r = fmod(a, b);

    if(r < 0.0)
    {
        r += b;
    }

    return r;
}

void run_case_ovr(
    const char *name,
    double **X,
    int *Y_class,
    int n_samples,
    int n_features,
    int n_classes,
    SVMKernelType kernel_type,
    double gamma,
    double degree,
    int max_iter)
{
    SVM_OVR *model =
        svm_ovr_create(
            n_samples,
            n_features,
            n_classes,
            kernel_type,
            gamma,
            degree,
            SVM_HARD_MARGIN); // cas de tests = hard margin pur, fidele au cours

    if(model == NULL)
    {
        fprintf(stderr, "Erreur: creation du modele impossible (%s).\n", name);
        return;
    }

    svm_ovr_train(
        model,
        X,
        Y_class,
        max_iter);

    int correct = 0;

    for(int i = 0; i < n_samples; i++)
    {
        if(svm_ovr_predict(model, X[i]) == Y_class[i])
        {
            correct++;
        }
    }

    double accuracy =
        (double)correct / (double)n_samples;

    printf(
        "[%-28s] accuracy = %.3f | n = %d (one-vs-rest, %d classifieurs)\n",
        name,
        accuracy,
        n_samples,
        n_classes);

    svm_ovr_free(model);
}

void run_case(
    const char *name,
    double **X,
    int *Y,
    int n_samples,
    int n_features,
    SVMKernelType kernel_type,
    double gamma,
    double degree,
    int max_iter)
{
    SVM *model =
        svm_create(
            n_samples,
            n_features,
            kernel_type,
            gamma,
            degree,
            SVM_HARD_MARGIN);

    if(model == NULL)
    {
        fprintf(stderr, "Erreur: creation du modele impossible (%s).\n", name);
        return;
    }

    svm_train(
        model,
        X,
        Y,
        max_iter);

    int correct = 0;

    for(int i = 0; i < n_samples; i++)
    {
        if(svm_predict(model, X[i]) == Y[i])
        {
            correct++;
        }
    }

    double accuracy =
        (double)correct / (double)n_samples;

    printf(
        "[%-28s] accuracy = %.3f | vecteurs supports = %d / %d | w0 = %.4f | iterations = %d\n",
        name,
        accuracy,
        svm_n_support_vectors(model),
        n_samples,
        model->b,
        svm_n_iterations_used(model));

    svm_free(model);
}

int main(void)
{
    {
        int n = 3, d = 2;
        double **X = alloc_matrix(n, d);
        int Y[3] = { 1, -1, -1 };

        X[0][0] = 1; X[0][1] = 1;
        X[1][0] = 2; X[1][1] = 3;
        X[2][0] = 3; X[2][1] = 3;

        run_case(
            "Linear Simple",
            X, Y, n, d,
            SVM_KERNEL_LINEAR, 0.0, 0.0,
            200000);

        free_matrix(X, n);
    }

    {
        srand(42);
        int n = 100, d = 2;
        double **X = alloc_matrix(n, d);
        int *Y = malloc(n * sizeof(int));

        for(int i = 0; i < 50; i++)
        {
            X[i][0] = randu() * 0.9 + 1.0;
            X[i][1] = randu() * 0.9 + 1.0;
            Y[i] = 1;
        }
        for(int i = 50; i < 100; i++)
        {
            X[i][0] = randu() * 0.9 + 2.0;
            X[i][1] = randu() * 0.9 + 2.0;
            Y[i] = -1;
        }

        run_case(
            "Linear Multiple",
            X, Y, n, d,
            SVM_KERNEL_LINEAR, 0.0, 0.0,
            200000);

        free_matrix(X, n);
        free(Y);
    }

    {
        int n = 4, d = 2;
        double **X = alloc_matrix(n, d);
        int Y[4] = { 1, 1, -1, -1 };

        X[0][0] = 1; X[0][1] = 0;
        X[1][0] = 0; X[1][1] = 1;
        X[2][0] = 0; X[2][1] = 0;
        X[3][0] = 1; X[3][1] = 1;

        run_case(
            "XOR (noyau RBF)",
            X, Y, n, d,
            SVM_KERNEL_RBF, 2.0, 0.0,
            200000);

        free_matrix(X, n);
    }

    {
        srand(42);
        int n = 200, d = 2;
        double **X = alloc_matrix(n, d);
        int *Y = malloc(n * sizeof(int));

        for(int i = 0; i < n; i++)
        {
            X[i][0] = randu() * 2.0 - 1.0;
            X[i][1] = randu() * 2.0 - 1.0;

            Y[i] =
                (fabs(X[i][0]) <= 0.3 || fabs(X[i][1]) <= 0.3)
                ? 1
                : -1;
        }

        run_case(
            "Cross (noyau RBF)",
            X, Y, n, d,
            SVM_KERNEL_RBF, 25.0, 0.0,
            200000);

        free_matrix(X, n);
        free(Y);
    }

    {
        srand(42);
        int n_gen = 500, d = 2;
        double **X_all = alloc_matrix(n_gen, d);
        int *Y_all = malloc(n_gen * sizeof(int));
        int n_valid = 0;

        for(int i = 0; i < n_gen; i++)
        {
            double p0 = randu() * 2.0 - 1.0;
            double p1 = randu() * 2.0 - 1.0;

            double f1 = -p0 - p1 - 0.5;
            double f2 =  p0 - p1 - 0.5;

            int cls = -1;

            if(f1 > 0 && p1 < 0 && f2 < 0)      cls = 0;
            else if(f1 < 0 && p1 > 0 && f2 < 0) cls = 1;
            else if(f1 < 0 && p1 < 0 && f2 > 0) cls = 2;

            if(cls != -1)
            {
                X_all[n_valid][0] = p0;
                X_all[n_valid][1] = p1;
                Y_all[n_valid] = cls;
                n_valid++;
            }
        }

        run_case_ovr(
            "Multi Linear 3 classes",
            X_all, Y_all, n_valid, d, 3,
            SVM_KERNEL_LINEAR, 0.0, 0.0,
            200000);

        free_matrix(X_all, n_gen);
        free(Y_all);
    }

    {
        srand(42);
        int n = 300, d = 2;
        double **X = alloc_matrix(n, d);
        int *Y = malloc(n * sizeof(int));

        for(int i = 0; i < n; i++)
        {
            double p0 = randu() * 2.0 - 1.0;
            double p1 = randu() * 2.0 - 1.0;

            X[i][0] = p0;
            X[i][1] = p1;

            int band0 = pymod(p0, 0.5) <= 0.25;
            int band1 = pymod(p1, 0.5) <= 0.25;

            if(band0 && !band1)      Y[i] = 0;
            else if(!band0 && band1) Y[i] = 1;
            else                     Y[i] = 2;
        }

        run_case_ovr(
            "Multi Cross",
            X, Y, n, d, 3,
            SVM_KERNEL_RBF, 30.0, 0.0,
            200000);

        free_matrix(X, n);
        free(Y);
    }

    return 0;
}