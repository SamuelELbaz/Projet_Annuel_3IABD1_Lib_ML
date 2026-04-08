/*
 * MODELE LINEAIRE SUPERVISE - ROSENBLATT vs PSEUDO-INVERSE
 * =========================================================
 *
 * ALGORITHME 1 - ROSENBLATT (Perceptron, 1957)
 *   - Prediction : classe = argmax_c ( W[:,c] . x )
 *   - Regle de mise a jour (seulement si erreur) :
 *       W[:, classe_vraie]   += alpha * x   (renforcer)
 *       W[:, classe_predite] -= alpha * x   (penaliser)
 *   - Converge SEULEMENT si les donnees sont lineairement separables
 *
 * ALGORITHME 2 - PSEUDO-INVERSE (solution analytique)
 *   - Formule : W = (X^T * X)^(-1) * X^T * Y
 *   - Minimise ||Y - X*W||^2 (moindres carres)
 *   - Solution optimale, calculee en une seule etape
 *   - Inversion par elimination de Gauss-Jordan
 *
 * Compilation (bibliotheque partagee) :
 *   gcc -shared -fPIC -DNO_MAIN -O2 -lm -o model_lineaire.so model_lineaire.c
 */

#include <math.h>
#include <string.h>

#define MAX_FEAT 5
#define MAX_CLS  3
#define MAX_SAMP 1000

/* Inversion de matrice NxN par Gauss-Jordan. Retourne 1 si OK, 0 si singuliere. */
int mat_inv(double A[MAX_FEAT][MAX_FEAT],
            double Ai[MAX_FEAT][MAX_FEAT], int n)
{
    double aug[MAX_FEAT][2 * MAX_FEAT];
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            aug[i][j]     = A[i][j];
            aug[i][n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < n; col++) {
        int piv = col;
        for (i = col + 1; i < n; i++)
            if (fabs(aug[i][col]) > fabs(aug[piv][col]))
                piv = i;

        if (fabs(aug[piv][col]) < 1e-12) {
            return 0;
        }

        for (j = 0; j < 2 * n; j++) {
            double tmp  = aug[col][j];
            aug[col][j] = aug[piv][j];
            aug[piv][j] = tmp;
        }

        double p = aug[col][col];
        for (j = 0; j < 2 * n; j++) aug[col][j] /= p;

        for (i = 0; i < n; i++) {
            if (i == col) continue;
            double f = aug[i][col];
            for (j = 0; j < 2 * n; j++)
                aug[i][j] -= f * aug[col][j];
        }
    }

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            Ai[i][j] = aug[i][n + j];

    return 1;
}

/* Prediction : classe = argmax_c (W[:,c] . x) */
int predict(double W[MAX_FEAT][MAX_CLS], double x[MAX_FEAT],
            int nf, int nc)
{
    int best = 0;
    double best_score = -1e30;
    for (int c = 0; c < nc; c++) {
        double s = 0.0;
        for (int k = 0; k < nf; k++) s += W[k][c] * x[k];
        if (s > best_score) { best_score = s; best = c; }
    }
    return best;
}

/* Rosenblatt (Perceptron) : mise a jour uniquement sur erreur */
void rosenblatt(double X[MAX_SAMP][MAX_FEAT], int Y[], int n,
                int nf, int nc, double alpha, int max_ep,
                double W[MAX_FEAT][MAX_CLS])
{
    for (int i = 0; i < MAX_FEAT; i++)
        for (int j = 0; j < MAX_CLS; j++)
            W[i][j] = 0.0;

    for (int ep = 0; ep < max_ep; ep++) {
        int errors = 0;

        for (int i = 0; i < n; i++) {
            int pred   = predict(W, X[i], nf, nc);
            int true_c = Y[i];

            if (pred != true_c) {
                errors++;
                for (int k = 0; k < nf; k++) {
                    W[k][true_c] += alpha * X[i][k];
                    W[k][pred]   -= alpha * X[i][k];
                }
            }
        }

        if (errors == 0) return;
    }
}

/* Pseudo-inverse : W = (X^T*X)^-1 * X^T*Y */
void pseudo_inverse(double X[MAX_SAMP][MAX_FEAT], int Y[], int n,
                    int nf, int nc, double W[MAX_FEAT][MAX_CLS])
{
    double Yoh[MAX_SAMP][MAX_CLS];
    memset(Yoh, 0, sizeof(Yoh));
    for (int i = 0; i < n; i++) Yoh[i][Y[i]] = 1.0;

    double XTX[MAX_FEAT][MAX_FEAT];
    memset(XTX, 0, sizeof(XTX));
    for (int i = 0; i < nf; i++)
        for (int j = 0; j < nf; j++)
            for (int k = 0; k < n; k++)
                XTX[i][j] += X[k][i] * X[k][j];

    double XTXi[MAX_FEAT][MAX_FEAT];
    memset(XTXi, 0, sizeof(XTXi));
    if (!mat_inv(XTX, XTXi, nf)) return;

    double XTY[MAX_FEAT][MAX_CLS];
    memset(XTY, 0, sizeof(XTY));
    for (int i = 0; i < nf; i++)
        for (int j = 0; j < nc; j++)
            for (int k = 0; k < n; k++)
                XTY[i][j] += X[k][i] * Yoh[k][j];

    for (int i = 0; i < nf; i++)
        for (int j = 0; j < nc; j++) {
            W[i][j] = 0.0;
            for (int k = 0; k < nf; k++)
                W[i][j] += XTXi[i][k] * XTY[k][j];
        }
}

/* Regression pseudo-inverse : W = (X^T*X + ridge*I)^-1 * X^T*y */
void pseudo_inverse_reg(double X[MAX_SAMP][MAX_FEAT], double Y_reg[MAX_SAMP],
                        int n, int nf, double ridge, double W[MAX_FEAT])
{
    double XTX[MAX_FEAT][MAX_FEAT];
    double XTXi[MAX_FEAT][MAX_FEAT];
    double XTY[MAX_FEAT];
    memset(XTX, 0, sizeof(XTX));
    memset(XTY, 0, sizeof(XTY));
    for (int i = 0; i < nf; i++)
        for (int j = 0; j < nf; j++)
            for (int k = 0; k < n; k++)
                XTX[i][j] += X[k][i] * X[k][j];
    for (int i = 0; i < nf; i++) XTX[i][i] += ridge;
    if (!mat_inv(XTX, XTXi, nf)) return;
    for (int i = 0; i < nf; i++)
        for (int k = 0; k < n; k++)
            XTY[i] += X[k][i] * Y_reg[k];
    for (int i = 0; i < nf; i++) {
        W[i] = 0.0;
        for (int k = 0; k < nf; k++)
            W[i] += XTXi[i][k] * XTY[k];
    }
}

/* Regression gradient descent (LMS) : W += alpha * err * x */
void gradient_descent(double X[MAX_SAMP][MAX_FEAT], double Y_reg[MAX_SAMP],
                      int n, int nf, double alpha, int max_ep, double W[MAX_FEAT])
{
    for (int i = 0; i < MAX_FEAT; i++) W[i] = 0.0;
    for (int ep = 0; ep < max_ep; ep++) {
        double mse = 0.0;
        for (int i = 0; i < n; i++) {
            double pred = 0.0;
            for (int k = 0; k < nf; k++) pred += W[k] * X[i][k];
            double err = Y_reg[i] - pred;
            mse += err * err;
            for (int k = 0; k < nf; k++)
                W[k] += alpha * err * X[i][k];
        }
        mse /= n;
        if (mse < 1e-10) return;
    }
}

