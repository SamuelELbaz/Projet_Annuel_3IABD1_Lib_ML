#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ===== InvMat.c =====
// ---- Allocation / libération ----
double **alloc_mat(int rows, int cols) {
    double **M = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) M[i] = calloc(cols, sizeof(double));
    return M;
}
void free_mat(double **M, int rows) {
    for (int i = 0; i < rows; i++) free(M[i]);
    free(M);
}

// ---- Inverse d'une matrice carrée n x n (Gauss-Jordan) ----
// Retourne 1 si OK, 0 si singulière. inv doit être pré-alloué.
int invert_square(double **a_in, double **inv, int n) {
    double **a = alloc_mat(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i][j]   = a_in[i][j];
            inv[i][j] = (i == j) ? 1.0 : 0.0;
        }

    for (int col = 0; col < n; col++) {
        // pivot partiel : on prend la plus grande valeur de la colonne
        int pivot = -1; double best = 1e-12;
        for (int row = col; row < n; row++)
            if (fabs(a[row][col]) > best) { best = fabs(a[row][col]); pivot = row; }
        if (pivot == -1) { free_mat(a, n); return 0; } // singulière

        // échange des lignes col <-> pivot
        double *tmp;
        tmp = a[col];   a[col]   = a[pivot];   a[pivot]   = tmp;
        tmp = inv[col]; inv[col] = inv[pivot]; inv[pivot] = tmp;

        double diag = a[col][col];
        for (int j = 0; j < n; j++) { a[col][j] /= diag; inv[col][j] /= diag; }

        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double f = a[row][col];
            for (int j = 0; j < n; j++) {
                a[row][j]   -= f * a[col][j];
                inv[row][j] -= f * inv[col][j];
            }
        }
    }
    free_mat(a, n);
    return 1;
}

// ---- Pseudo-inverse de Moore-Penrose ----
// A : matrice m x n. Retourne A+ de taille n x m (à libérer avec free_mat(., n)),
// ou NULL si A^T A (resp. A A^T) est singulière.
double **pseudo_inverse(double **A, int m, int n) {
    double **Aplus = alloc_mat(n, m);

    if (m >= n) {
        // A+ = (A^T A)^-1 A^T
        double **AtA = alloc_mat(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < m; k++)
                    AtA[i][j] += A[k][i] * A[k][j];

        double **inv = alloc_mat(n, n);
        if (!invert_square(AtA, inv, n)) {
            free_mat(AtA, n); free_mat(inv, n); free_mat(Aplus, n);
            return NULL;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                for (int k = 0; k < n; k++)
                    Aplus[i][j] += inv[i][k] * A[j][k]; // (A^T)[k][j] = A[j][k]

        free_mat(AtA, n); free_mat(inv, n);
    } else {
        // A+ = A^T (A A^T)^-1
        double **AAt = alloc_mat(m, m);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                for (int k = 0; k < n; k++)
                    AAt[i][j] += A[i][k] * A[j][k];

        double **inv = alloc_mat(m, m);
        if (!invert_square(AAt, inv, m)) {
            free_mat(AAt, m); free_mat(inv, m); free_mat(Aplus, n);
            return NULL;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                for (int k = 0; k < m; k++)
                    Aplus[i][j] += A[k][i] * inv[k][j]; // (A^T)[i][k] = A[k][i]

        free_mat(AAt, m); free_mat(inv, m);
    }
    return Aplus;
}
// ====================

// Helpeur
double** allocation_matricielle(int largeur, int hauteur){
    double** m = malloc(largeur * sizeof(double*));
    for(int i = 0; i < largeur; i++){
        m[i] = calloc(hauteur, sizeof(double));
    }
    return m;
}

void afficher_matrice(const char* nom, double** M, int lignes, int colonnes){
    printf("=== %s ===\n", nom);
    for(int i = 0; i < lignes; i++){
        for(int j = 0; j < colonnes; j++) printf("% .4f ", M[i][j]);
        printf("\n");
    }
    printf("\n");
}

// == RBF ==
double distance_quadratique(double* a, double* b, double dimension){
    double sum = 0;
    for(int i = 0; i < dimension; i++){
        double distance = a[i] - b[i];
        sum += distance*distance;
    }
    return sum;
}

//𝑒(−𝛾(||𝑋1−𝑋2|)^2)
double influence(double* a, double* b, double gamma, double dimension){
    return exp(-gamma * (distance_quadratique(a, b, dimension)));
}

double** remplissage_matricielle(double** X, double taille, int dimension, double gamma){
    double** matrice = allocation_matricielle(taille, taille);
    for(int i = 0; i < taille; i++){
        for(int j = 0; j < taille; j++){
            matrice[i][j] = influence(X[i], X[j], gamma, dimension);
        }
    }
    return matrice;
}

 // PROTOCOLE_DE_MATRIFICATION_DES_MASSES_NUMERIQUES
// W[k][i] = poids du centre i pour la classe k
double** calcul_poids(double** matInv, double** y_true, int nb_centre, int nb_classes){
    double** w = allocation_matricielle(nb_classes, nb_centre);
    for (int classe = 0; classe < nb_classes; classe++){
        for (int centre = 0; centre < nb_centre; centre++){
            for (int exemple = 0; exemple < nb_centre; exemple++){
                w[classe][centre] += matInv[centre][exemple] * y_true[exemple][classe];
            }
        }
    }
    return w;
}

double** k_mean(double** mat,int nb_point, int dimension, int nb_centre){
    double** centre  = allocation_matricielle(nb_centre, dimension);
    int* assignation      = malloc(nb_point * sizeof(int));
    int* assignation_prec = malloc(nb_point * sizeof(int));
    
    int max_iter = 100;
    
     // Init des centres -> prends des points random comme coord de centre
    // Todo : les espacer ou prendre des points de chaque classe
    for (int i = 0; i < nb_centre; i++){
        int rd = rand() % nb_point;
        for (int j = 0; j < dimension; j++){
            centre[i][j] = mat[rd][j];
        }
    }
    for (int i = 0; i < nb_point; i++) assignation_prec[i] = -1;
    
    // Sacro-Saibte boucle Assignation-Udpate
    for (int iter = 0; iter < max_iter; iter++){
        
        // Assignation initial des points a un centre
        for(int i = 0; i < nb_point; i++){
            int best_centre = -1;
            double best_dist = INFINITY;
            
            for(int j = 0; j < nb_centre; j++){
                double dist = distance_quadratique(mat[i], centre[j], dimension);
                if (dist < best_dist){
                    best_dist = dist;
                    best_centre = j;
                }
            }
            assignation[i] = best_centre;
        }
        
        // CONDITIONNAGE SOVIETIQUE DE L'ARRET
        int change = 0;
        for (int i = 0; i < nb_point; i++){
            if (assignation[i] != assignation_prec[i]){ change = 1; break; }
        }
        if (!change) break;
        
        // MISE A JOUR INDUSTRIELLE DE LA POSITION DES CENTRES
        for (int i = 0; i < nb_centre; i++){
            double* som_pos_point = calloc(dimension, sizeof(double));
            int nb_point_in_cluster = 0;
            
            for (int j = 0; j < nb_point; j++){
                if (assignation[j] == i){
                    nb_point_in_cluster += 1;
                    for (int k = 0; k < dimension; k++){
                        som_pos_point[k] += mat[j][k];
                    }
                }
            }
            
            if (nb_point_in_cluster != 0){
                for (int j = 0; j < dimension; j++){
                    centre[i][j] = som_pos_point[j] / nb_point_in_cluster;
                } 
            }
            free(som_pos_point);
        }
        
        // Pour condition d'arret au debut du prochain tour
        memcpy(assignation_prec, assignation, nb_point * sizeof(int));
    }
    
    free(assignation);
    free(assignation_prec);
    return centre;
}

int main(){
    srand((unsigned int)time(NULL));
    
    double p0[] = {5.0, 2.0, 9.0};
    double p1[] = {2.0, 6.0, 3.0};
    double p2[] = {4.0, 4.0, 5.0};
    double p3[] = {6.0, 1.0, 2.0};
    double p4[] = {3.0, 8.0, 7.0};
    double p5[] = {9.0, 2.0, 1.0};
    double* X[] = {p0, p1, p2, p3, p4, p5};

    // Y one-hot : p0->classe0, p1->classe0, p2->classe1
    // y_true[exemple][classe]  (matrice N x K)
    double y0[] = {1, 0, 0};
    double y1[] = {1, 0, 0};
    double y2[] = {0, 1, 0};
    double y3[] = {0, 1, 0};
    double y4[] = {0, 0, 1};
    double y5[] = {0, 0, 1};
    double* Y[] = {y0, y1, y2, y3, y4, y5};

    int nb_centre  = 6;
    int nb_classes = 3;
    int dimension  = 3;
    double gamma   = 0.03;

    double** mat    = remplissage_matricielle(X, nb_centre, dimension, gamma);
    afficher_matrice("Mat", mat, nb_centre, nb_centre);

    double** matInv = pseudo_inverse(mat, nb_centre, nb_centre);
    afficher_matrice("Mat_INV", matInv, nb_centre, nb_centre);

    double** W = calcul_poids(matInv, Y, nb_centre, nb_classes);
    afficher_matrice("W (nb_classes x nb_centre)", W, nb_classes, nb_centre);

    printf("=== PREDICTIONS (score par classe) ===\n");
    for(int i = 0; i < nb_centre; i++){
        printf("exemple %d : ", i);
        for(int classe = 0; classe < nb_classes; classe++){
            double score = 0.0;
            for(int j = 0; j < nb_centre; j++)
                score += W[classe][j] * influence(X[i], X[j], gamma, dimension);
            printf("cl%d=% .4f  ", classe, score);
        }
        printf("\n");
    }
    
    printf("======================\n");
    int nb_centre_kmean = 4;
    double** k = k_mean(X, nb_centre, dimension, nb_centre_kmean);
    afficher_matrice("Init K Centre", k, nb_centre_kmean, dimension);
    
    return 0;
}
