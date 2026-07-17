#include <stdlib.h>
#include <string.h>

// Acces row-major : ligne i, colonne j
#define X_AT(i, j) X[(i) * nf + (j)]
#define W_AT(i, j) W[(i) * nc + (j)]
#define Y_AT(i, j) Y[(i) * nc + (j)]

static void shuffle(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

static int predict_internal(double *W, double *x, int nf, int nc) {
    int best = 0;
    double best_score = -1e30;
    for (int c = 0; c < nc; c++) {
        double s = 0.0;
        for (int k = 0; k < nf; k++) s += W[k * nc + c] * x[k];
        if (s > best_score) { best_score = s; best = c; }
    }
    return best;
}

// Perceptron de Rosenblatt (classification)
void rosenblatt(double *X, int *Y, int n, int nf, int nc,
                double alpha, int max_ep, double *W, unsigned int seed)
{
    memset(W, 0, nf * nc * sizeof(double));

    int *indices = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;
    srand(seed);  // seed passe depuis Python, pour pouvoir reproduire un run

    for (int ep = 0; ep < max_ep; ep++) {
        shuffle(indices, n);
        int errors = 0;
        for (int ii = 0; ii < n; ii++) {
            int i = indices[ii];
            double *xi = X + i * nf;
            int pred = predict_internal(W, xi, nf, nc);
            int true_c = Y[i];
            if (pred != true_c) {
                errors++;
                for (int k = 0; k < nf; k++) {
                    W[k * nc + true_c] += alpha * xi[k];
                    W[k * nc + pred]   -= alpha * xi[k];
                }
            }
        }
        if (errors == 0) break;
    }
    free(indices);
}

// Inversion de matrice par Gauss-Jordan (in place).
// M et inv font size x size. inv doit deja contenir l'identite en entree.
// Retourne 0 si M est singuliere (pas inversible), 1 sinon.
static int invert_matrix(double *M, int size, double *inv)
{
    for (int col = 0; col < size; col++) {
        // on cherche le meilleur pivot dans la colonne (plus stable numeriquement)
        int pivot = col;
        for (int row = col + 1; row < size; row++) {
            double a = M[row * size + col];
            double b = M[pivot * size + col];
            if (a < 0) a = -a;
            if (b < 0) b = -b;
            if (a > b) pivot = row;
        }

        double diag = M[pivot * size + col];
        if (diag == 0.0 || diag != diag) return 0;  // matrice singuliere ou NaN

        if (pivot != col) {
            for (int k = 0; k < size; k++) {
                double tmp = M[col * size + k];
                M[col * size + k] = M[pivot * size + k];
                M[pivot * size + k] = tmp;
                tmp = inv[col * size + k];
                inv[col * size + k] = inv[pivot * size + k];
                inv[pivot * size + k] = tmp;
            }
        }

        diag = M[col * size + col];
        for (int k = 0; k < size; k++) {
            M[col * size + k] /= diag;
            inv[col * size + k] /= diag;
        }

        for (int row = 0; row < size; row++) {
            if (row == col) continue;
            double factor = M[row * size + col];
            for (int k = 0; k < size; k++) {
                M[row * size + k] -= factor * M[col * size + k];
                inv[row * size + k] -= factor * inv[col * size + k];
            }
        }
    }
    return 1;
}

void pseudo_inverse(double *X, double *Y, int n, int nf, int nc, double *W)
{
    memset(W, 0, nf * nc * sizeof(double));

    if (nf <= n) {
        double *XtX = (double *)calloc(nf * nf, sizeof(double));
        double *XtY = (double *)calloc(nf * nc, sizeof(double));
        double *inv = (double *)calloc(nf * nf, sizeof(double));
        for (int i = 0; i < nf; i++) inv[i * nf + i] = 1.0;

        for (int i = 0; i < nf; i++)
            for (int j = 0; j < nf; j++)
                for (int k = 0; k < n; k++)
                    XtX[i * nf + j] += X[k * nf + i] * X[k * nf + j];

        for (int i = 0; i < nf; i++)
            for (int j = 0; j < nc; j++)
                for (int k = 0; k < n; k++)
                    XtY[i * nc + j] += X[k * nf + i] * Y[k * nc + j];

        if (invert_matrix(XtX, nf, inv)) {
            for (int i = 0; i < nf; i++)
                for (int j = 0; j < nc; j++)
                    for (int k = 0; k < nf; k++)
                        W[i * nc + j] += inv[i * nf + k] * XtY[k * nc + j];
        }

        free(XtX); free(XtY); free(inv);

    } else {
        // Forme duale : plus de features que d'exemples, Xt X serait singuliere.
        // On passe par X Xt (n x n), beaucoup plus petit et inversible.
        double *XXt = (double *)calloc(n * n, sizeof(double));
        double *inv = (double *)calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++) inv[i * n + i] = 1.0;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < nf; k++)
                    XXt[i * n + j] += X[i * nf + k] * X[j * nf + k];

        if (invert_matrix(XXt, n, inv)) {
            // alpha = (X Xt)^-1 Y   (taille n x nc)
            double *alpha = (double *)calloc(n * nc, sizeof(double));
            for (int i = 0; i < n; i++)
                for (int j = 0; j < nc; j++)
                    for (int k = 0; k < n; k++)
                        alpha[i * nc + j] += inv[i * n + k] * Y[k * nc + j];

            // W = Xt . alpha   (taille nf x nc)
            for (int i = 0; i < nf; i++)
                for (int j = 0; j < nc; j++)
                    for (int k = 0; k < n; k++)
                        W[i * nc + j] += X[k * nf + i] * alpha[k * nc + j];

            free(alpha);
        }

        free(XXt); free(inv);
    }
}

int predict(double *W, double *x, int nf, int nc) {
    return predict_internal(W, x, nf, nc);
}

double predict_regression(double *W, double *x, int nf, int output_idx, int nc) {
    double s = 0.0;
    for (int k = 0; k < nf; k++)
        s += W[k * nc + output_idx] * x[k];
    return s;
}
