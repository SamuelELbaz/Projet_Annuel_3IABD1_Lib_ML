#ifndef SVM_H
#define SVM_H

typedef enum
{
    SVM_KERNEL_LINEAR = 0, // K(x,y) = x.y
    SVM_KERNEL_POLY   = 1, // K(x,y) = (1 + x.y)^degree
    SVM_KERNEL_RBF    = 2  // K(x,y) = exp(-gamma * ||x-y||^2)

} SVMKernelType;

typedef struct
{
    SVMKernelType kernel_type;
    double gamma;
    double degree;

    double *alpha;

    double **X;
    int *Y;

    double *w;

    double b;

    double C_soft;

    int n_samples;
    int n_features;

    int n_iter_used;

    int owns_data;

} SVM;

#define SVM_HARD_MARGIN 1e30

SVM* svm_create(
    int n_samples,
    int n_features,
    SVMKernelType kernel_type,
    double gamma,
    double degree,
    double C_soft);

void svm_free(
    SVM *model);

void svm_train(
    SVM *model,
    double **X,
    int *Y,
    int max_iter);

int svm_n_iterations_used(
    SVM *model);

int svm_save(
    SVM *model,
    const char *path);

SVM* svm_load(
    const char *path);

double svm_kernel(
    SVM *model,
    double *a,
    double *b);

double svm_score(
    SVM *model,
    double *x);

int svm_predict(
    SVM *model,
    double *x);

double svm_norme_dans_espace_noyau(
    SVM *model);

int svm_n_support_vectors(
    SVM *model);

typedef struct
{
    SVM **classifiers;
    int **Y_per_class;
    int n_classes;
    int n_samples;
    int n_features;
} SVM_OVR;

SVM_OVR* svm_ovr_create(
    int n_samples,
    int n_features,
    int n_classes,
    SVMKernelType kernel_type,
    double gamma,
    double degree,
    double C_soft);

void svm_ovr_free(
    SVM_OVR *model);

void svm_ovr_train(
    SVM_OVR *model,
    double **X,
    int *Y_class,
    int max_iter);

int svm_ovr_predict(
    SVM_OVR *model,
    double *x);

int svm_ovr_n_support_vectors(
    SVM_OVR *model,
    int class_idx);

int svm_ovr_n_iterations_used(
    SVM_OVR *model,
    int class_idx);

int svm_ovr_get_n_classes(
    SVM_OVR *model);

int svm_ovr_save(
    SVM_OVR *model,
    const char *path);

SVM_OVR* svm_ovr_load(
    const char *path);


#endif
