#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "svm.h"

double dot(
    double *a,
    double *b,
    int size)
{
    double result = 0.0;

    for(int i = 0; i < size; i++)
    {
        result += a[i] * b[i];
    }

    return result;
}

double sq_dist(
    double *a,
    double *b,
    int size)
{
    double result = 0.0;

    for(int i = 0; i < size; i++)
    {
        double d = a[i] - b[i];
        result += d * d;
    }

    return result;
}

double svm_kernel(
    SVM *model,
    double *a,
    double *b)
{
    switch(model->kernel_type)
    {
        case SVM_KERNEL_POLY:
            return pow(
                1.0 + dot(a, b, model->n_features),
                model->degree);

        case SVM_KERNEL_RBF:
            return exp(
                -model->gamma
                * sq_dist(a, b, model->n_features));

        case SVM_KERNEL_LINEAR:
        default:
            return dot(a, b, model->n_features);
    }
}

SVM* svm_create(
    int n_samples,
    int n_features,
    SVMKernelType kernel_type,
    double gamma,
    double degree,
    double C_soft)
{
    SVM *model =
        (SVM*)malloc(sizeof(SVM));

    if(model == NULL)
    {
        return NULL;
    }

    model->n_samples = n_samples;
    model->n_features = n_features;

    model->b = 0.0;
    model->C_soft = C_soft;
    model->n_iter_used = 0;
    model->owns_data = 0;

    model->kernel_type = kernel_type;
    model->gamma = gamma;
    model->degree = degree;

    model->X = NULL;
    model->Y = NULL;

    model->alpha =
        (double*)calloc(
            n_samples,
            sizeof(double));

    model->w =
        (double*)calloc(
            n_features,
            sizeof(double));

    if(model->alpha == NULL || model->w == NULL)
    {
        free(model->alpha);
        free(model->w);
        free(model);
        return NULL;
    }

    return model;
}

void svm_free(
    SVM *model)
{
    if(model == NULL)
    {
        return;
    }

    if(model->owns_data)
    {
        if(model->X != NULL)
        {
            for(int i = 0; i < model->n_samples; i++)
            {
                free(model->X[i]);
            }
            free(model->X);
        }
        free(model->Y);
    }

    free(model->alpha);
    free(model->w);

    free(model);
}

void compute_w(
    SVM *model)
{
    if(model->kernel_type != SVM_KERNEL_LINEAR)
    {
        return;
    }

    for(int j = 0; j < model->n_features; j++)
    {
        model->w[j] = 0.0;

        for(int i = 0; i < model->n_samples; i++)
        {
            model->w[j] +=
                model->alpha[i]
                * model->Y[i]
                * model->X[i][j];
        }
    }
}

double* build_Q(
    SVM *model,
    double **X,
    int *Y)
{
    int n = model->n_samples;

    double *Q =
        malloc((size_t)n * (size_t)n * sizeof(double));

    for(int i = 0; i < n; i++)
    {
        for(int j = i; j < n; j++)
        {
            double value =
                (double)Y[i] * (double)Y[j]
                * svm_kernel(model, X[i], X[j]);

            Q[i * n + j] = value;
            Q[j * n + i] = value;
        }
    }

    return Q;
}

void project_onto_constraints(
    double *v,
    int *Y,
    int n,
    double C,
    double *alpha_out)
{
    double theta_lo = -1e6;
    double theta_hi = 1e6;

    for(int iter = 0; iter < 100; iter++)
    {
        double theta = (theta_lo + theta_hi) / 2.0;
        double g = 0.0;

        for(int i = 0; i < n; i++)
        {
            double a = v[i] - (double)Y[i] * theta;
            if(a < 0.0) a = 0.0;
            if(a > C) a = C;
            g += (double)Y[i] * a;
        }

        if(g > 0.0)
        {
            theta_lo = theta;
        }
        else
        {
            theta_hi = theta;
        }
    }

    double theta = (theta_lo + theta_hi) / 2.0;

    for(int i = 0; i < n; i++)
    {
        double a = v[i] - (double)Y[i] * theta;
        if(a < 0.0) a = 0.0;
        if(a > C) a = C;
        alpha_out[i] = a;
    }
}

void svm_train(
    SVM *model,
    double **X,
    int *Y,
    int max_iter)
{
    model->X = X;
    model->Y = Y;

    int n = model->n_samples;

    double *Q = build_Q(model, X, Y);

    double q_max_diag = 1e-9;
    for(int i = 0; i < n; i++)
    {
        if(Q[i * n + i] > q_max_diag)
        {
            q_max_diag = Q[i * n + i];
        }
    }
    double eta = 1.0 / (q_max_diag * (double)n);

    double *grad = malloc((size_t)n * sizeof(double));
    double *v = malloc((size_t)n * sizeof(double));
    double *alpha_prev = malloc((size_t)n * sizeof(double));

    const double CONV_TOL_REL = 1e-3;
    const double CONV_TOL = CONV_TOL_REL * eta;
    const int CONV_PATIENCE = 20;
    int stable_count = 0;
    int iter;

    for(iter = 0; iter < max_iter; iter++)
    {
        memcpy(alpha_prev, model->alpha, (size_t)n * sizeof(double));

        for(int i = 0; i < n; i++)
        {
            double s = 0.0;
            for(int j = 0; j < n; j++)
            {
                s += Q[i * n + j] * model->alpha[j];
            }
            grad[i] = s - 1.0;
        }

        for(int i = 0; i < n; i++)
        {
            v[i] = model->alpha[i] - eta * grad[i];
        }

        project_onto_constraints(v, Y, n, model->C_soft, model->alpha);

        double max_delta = 0.0;
        for(int i = 0; i < n; i++)
        {
            double delta = fabs(model->alpha[i] - alpha_prev[i]);
            if(delta > max_delta)
            {
                max_delta = delta;
            }
        }

        if(max_delta < CONV_TOL)
        {
            stable_count++;
            if(stable_count >= CONV_PATIENCE)
            {
                iter++;
                break;
            }
        }
        else
        {
            stable_count = 0;
        }
    }

    model->n_iter_used = iter;

    free(grad);
    free(v);
    free(alpha_prev);
    free(Q);

    compute_w(model);

    int sv_index = -1;
    double best_alpha = 1e-6;

    for(int i = 0; i < n; i++)
    {
        if(model->alpha[i] > best_alpha)
        {
            best_alpha = model->alpha[i];
            sv_index = i;
        }
    }

    if(sv_index == -1)
    {
        model->b = 0.0;
        return;
    }

    double sum = 0.0;
    for(int m = 0; m < n; m++)
    {
        if(model->alpha[m] <= 0.0)
        {
            continue;
        }
        sum +=
            model->alpha[m] * (double)Y[m]
            * svm_kernel(model, X[m], X[sv_index]);
    }

    model->b = (1.0 / (double)Y[sv_index]) - sum;
}

double svm_score(
    SVM *model,
    double *x)
{
    double result = model->b;

    for(int i = 0; i < model->n_samples; i++)
    {
        if(model->alpha[i] <= 0.0)
        {
            continue;
        }

        result +=
            model->alpha[i]
            * model->Y[i]
            * svm_kernel(model, model->X[i], x);
    }

    return result;
}

int svm_predict(
    SVM *model,
    double *x)
{
    double score = svm_score(model, x);

    if(score >= 0.0)
    {
        return 1;
    }

    return -1;
}

double svm_norme_dans_espace_noyau(
    SVM *model)
{
    double somme = 0.0;

    for(int i = 0; i < model->n_samples; i++)
    {
        for(int j = 0; j < model->n_samples; j++)
        {
            somme += model->alpha[i] * model->alpha[j]
                * (double)model->Y[i] * (double)model->Y[j]
                * svm_kernel(model, model->X[i], model->X[j]);
        }
    }

    return sqrt(somme > 0.0 ? somme : 0.0);
}

int svm_n_support_vectors(
    SVM *model)
{
    int count = 0;

    for(int i = 0; i < model->n_samples; i++)
    {
        if(model->alpha[i] > 1e-6)
        {
            count++;
        }
    }

    return count;
}

int svm_n_iterations_used(
    SVM *model)
{
    return model->n_iter_used;
}

static void svm_write_to_fp(
    SVM *model,
    FILE *f)
{
    int n_sv = svm_n_support_vectors(model);

    fprintf(f, "%d %d %d\n", (int)model->kernel_type, model->n_features, n_sv);
    fprintf(f, "%.17g %.17g %.17g\n", model->gamma, model->degree, model->b);

    for(int i = 0; i < model->n_samples; i++)
    {
        if(model->alpha[i] <= 1e-6)
        {
            continue;
        }

        fprintf(f, "%.17g %d", model->alpha[i], model->Y[i]);
        for(int j = 0; j < model->n_features; j++)
        {
            fprintf(f, " %.17g", model->X[i][j]);
        }
        fprintf(f, "\n");
    }
}

static SVM* svm_read_from_fp(
    FILE *f)
{
    int kernel_type_int, n_features, n_sv;

    if(fscanf(f, "%d %d %d", &kernel_type_int, &n_features, &n_sv) != 3)
    {
        return NULL;
    }

    double gamma, degree, b;

    if(fscanf(f, "%lf %lf %lf", &gamma, &degree, &b) != 3)
    {
        return NULL;
    }

    SVM *model = svm_create(n_sv, n_features, (SVMKernelType)kernel_type_int, gamma, degree, SVM_HARD_MARGIN);

    if(model == NULL)
    {
        return NULL;
    }

    model->b = b;
    model->owns_data = 1;

    model->X = (double**)malloc((size_t)n_sv * sizeof(double*));
    model->Y = (int*)malloc((size_t)n_sv * sizeof(int));

    for(int i = 0; i < n_sv; i++)
    {
        model->X[i] = (double*)malloc((size_t)n_features * sizeof(double));

        if(fscanf(f, "%lf %d", &model->alpha[i], &model->Y[i]) != 2)
        {
            svm_free(model);
            return NULL;
        }

        for(int j = 0; j < n_features; j++)
        {
            if(fscanf(f, "%lf", &model->X[i][j]) != 1)
            {
                svm_free(model);
                return NULL;
            }
        }
    }

    compute_w(model);

    return model;
}


int svm_save(
    SVM *model,
    const char *path)
{
    FILE *f = fopen(path, "w");

    if(f == NULL)
    {
        return -1;
    }

    svm_write_to_fp(model, f);

    fclose(f);
    return 0;
}

SVM* svm_load(
    const char *path)
{
    FILE *f = fopen(path, "r");

    if(f == NULL)
    {
        return NULL;
    }

    SVM *model = svm_read_from_fp(f);

    fclose(f);
    return model;
}

// SVM multi-classes one-vs-rest (cf. svm.h pour le principe).
// Reutilise simplement svm_create / svm_train / svm_score / svm_free
// ci-dessus, K fois (un SVM binaire par classe).

SVM_OVR* svm_ovr_create(
    int n_samples,
    int n_features,
    int n_classes,
    SVMKernelType kernel_type,
    double gamma,
    double degree,
    double C_soft)
{
    SVM_OVR *model =
        (SVM_OVR*)malloc(sizeof(SVM_OVR));

    if(model == NULL)
    {
        return NULL;
    }

    model->n_samples = n_samples;
    model->n_features = n_features;
    model->n_classes = n_classes;

    model->classifiers =
        malloc(n_classes * sizeof(SVM*));

    model->Y_per_class =
        malloc(n_classes * sizeof(int*));

    for(int c = 0; c < n_classes; c++)
    {
        model->classifiers[c] =
            svm_create(
                n_samples,
                n_features,
                kernel_type,
                gamma,
                degree,
                C_soft);

        model->Y_per_class[c] =
            malloc(n_samples * sizeof(int));
    }

    return model;
}

void svm_ovr_free(
    SVM_OVR *model)
{
    if(model == NULL)
    {
        return;
    }

    for(int c = 0; c < model->n_classes; c++)
    {
        svm_free(model->classifiers[c]);

        if(model->Y_per_class != NULL)
        {
            free(model->Y_per_class[c]);
        }
    }

    free(model->classifiers);
    free(model->Y_per_class);
    free(model);
}

void svm_ovr_train(
    SVM_OVR *model,
    double **X,
    int *Y_class,
    int max_iter)
{
    for(int c = 0; c < model->n_classes; c++)
    {
        for(int i = 0; i < model->n_samples; i++)
        {
            model->Y_per_class[c][i] =
                (Y_class[i] == c)
                ? 1
                : -1;
        }

        svm_train(
            model->classifiers[c],
            X,
            model->Y_per_class[c],
            max_iter);
    }
}

int svm_ovr_predict(
    SVM_OVR *model,
    double *x)
{
    int best_class = 0;
    double best_score =
        svm_score(model->classifiers[0], x);

    for(int c = 1; c < model->n_classes; c++)
    {
        double score =
            svm_score(model->classifiers[c], x);

        if(score > best_score)
        {
            best_score = score;
            best_class = c;
        }
    }

    return best_class;
}

int svm_ovr_n_support_vectors(
    SVM_OVR *model,
    int class_idx)
{
    return svm_n_support_vectors(model->classifiers[class_idx]);
}

int svm_ovr_n_iterations_used(
    SVM_OVR *model,
    int class_idx)
{
    return svm_n_iterations_used(model->classifiers[class_idx]);
}

int svm_ovr_get_n_classes(
    SVM_OVR *model)
{
    return model->n_classes;
}

int svm_ovr_save(
    SVM_OVR *model,
    const char *path)
{
    FILE *f = fopen(path, "w");

    if(f == NULL)
    {
        return -1;
    }

    fprintf(f, "%d\n", model->n_classes);

    for(int c = 0; c < model->n_classes; c++)
    {
        svm_write_to_fp(model->classifiers[c], f);
    }

    fclose(f);
    return 0;
}

SVM_OVR* svm_ovr_load(
    const char *path)
{
    FILE *f = fopen(path, "r");

    if(f == NULL)
    {
        return NULL;
    }

    int n_classes;

    if(fscanf(f, "%d", &n_classes) != 1)
    {
        fclose(f);
        return NULL;
    }

    SVM_OVR *model = (SVM_OVR*)malloc(sizeof(SVM_OVR));
    model->n_classes = n_classes;
    model->classifiers = (SVM**)malloc((size_t)n_classes * sizeof(SVM*));
    model->Y_per_class = NULL;  // non utilise pour un modele charge (deja entraine)
    model->n_samples = 0;
    model->n_features = 0;

    for(int c = 0; c < n_classes; c++)
    {
        model->classifiers[c] = svm_read_from_fp(f);

        if(model->classifiers[c] == NULL)
        {
            fclose(f);
            svm_ovr_free(model);
            return NULL;
        }

        model->n_features = model->classifiers[c]->n_features;
    }

    fclose(f);
    return model;
}
