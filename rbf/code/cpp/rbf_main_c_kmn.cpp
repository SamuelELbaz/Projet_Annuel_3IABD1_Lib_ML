#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "eigen-5.0.0/Eigen/Dense"

extern "C" { 
void set_seed(unsigned int s){ srand(s); }

// == Header magique (flemme de faire un header séparé)
double** allocation_matricielle(int largeur, int hauteur);
void liberateur_judiciaire(double** mat, int rows);

// == Inversion Matricielles ==

// Eigen
double** pseudo_inverse(double** A, int m, int n) {
    Eigen::MatrixXd Mat(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            Mat(i, j) = A[i][j];
 
    Eigen::MatrixXd Pinv = Mat.completeOrthogonalDecomposition().pseudoInverse();

    double** Aplus = allocation_matricielle(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            Aplus[i][j] = Pinv(i, j);
 
    return Aplus;
}

// ============================


// Helpeur
double** allocation_matricielle(int largeur, int hauteur){
    double** m = (double**)malloc(largeur * sizeof(double*));
    for(int i = 0; i < largeur; i++){
        m[i] = (double*)calloc(hauteur, sizeof(double));
    }
    return m;
}

void liberateur_judiciaire(double** mat, int rows) {
    for (int i = 0; i < rows; i++) free(mat[i]);
    free(mat);
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

// Remplit la matrice d'influence pour 1 exemple = 1 point
double** remplissage_matricielle(double** X, double taille, int dimension, double gamma){
    double** matrice = allocation_matricielle(taille, taille);
    for(int i = 0; i < taille; i++){
        for(int j = 0; j < taille; j++){
            matrice[i][j] = influence(X[i], X[j], gamma, dimension);
        }
    }
    return matrice;
}

// Remplit la matrice d'influence -> influence des centres sur les points
double** remplissage_matricielle_centre(double** points, int nb_points, 
                                        double** centres, int nb_centres, 
                                        int dimension, double gamma){
    double** matrice = allocation_matricielle(nb_points, nb_centres);
    for (int i = 0; i < nb_points; i++){
        for (int j = 0; j < nb_centres; j++){
            matrice[i][j] = influence(points[i], centres[j], gamma, dimension);
        }
    }
    return matrice;
}

 // PROTOCOLE_DE_MATRIFICATION_DES_MASSES_NUMERIQUES
// W[k][i] = poids du centre i pour la classe k
double** calcul_poids(double** matInv, double** y_true, int nb_centre, int nb_point, int nb_classes){
    double** w = allocation_matricielle(nb_classes, nb_centre);
    for (int classe = 0; classe < nb_classes; classe++){
        for (int centre = 0; centre < nb_centre; centre++){
            for (int exemple = 0; exemple < nb_point; exemple++){
                w[classe][centre] += matInv[centre][exemple] * y_true[exemple][classe];
            }
        }
    }
    return w;
}

double** k_mean(double** mat,int nb_point, int dimension, int nb_centre){
    double** centre  = allocation_matricielle(nb_centre, dimension);
    int* assignation      = (int*)malloc(nb_point * sizeof(int));
    int* assignation_prec = (int*)malloc(nb_point * sizeof(int));
    
    double* dist = (double*)malloc(nb_point * sizeof(double));
    
    int max_iter = 100;
    
    /**/
    // Init des centres -> prends des points random comme coord de centre
    for (int i = 0; i < nb_centre; i++){
        int rd = rand() % nb_point;
        for (int j = 0; j < dimension; j++){
            centre[i][j] = mat[rd][j];
        }
    }
    
    
    // Init des centres -> Eloignés les uns des autres
    /*
    int rd = rand() % nb_point;
    for (int i = 0; i < dimension; i++){
        centre[0][i] = mat[rd][i];
    }
    
    for (int i = 1; i < nb_centre; i++){
        for (int p = 0; p < nb_point; p++){
            double best = INFINITY;
            for (int c = 0; c < i; c++){
                double d = distance_quadratique(mat[p], centre[c], dimension);
                if (d < best) best = d;
            }
            dist[p] = best;
        }
        
        double total = 0;
        for (int p = 0; p < nb_point; p++) total += dist[p];
        
        double r = ((double)rand() / RAND_MAX) * total;
    
        int elu = nb_point - 1;
        double cumul = 0;
        for (int p = 0; p < nb_point; p++){
            cumul += dist[p];
            if (cumul >= r){ elu = p; break; }
        }
    
        for (int j = 0; j < dimension; j++)
        centre[i][j] = mat[elu][j];
    }
    */
    
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
            double* som_pos_point = (double*)calloc(dimension, sizeof(double));
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
    free(dist);
    return centre;
}

// Utilisation en lib

double** unflat(double* X_flat, int largeur, int hauteur){
    double** X_unflated = (double**)malloc(hauteur * sizeof(double*));
    if (!X_unflated)
        return NULL;

    for (int i = 0; i < hauteur; i++)
        X_unflated[i] = X_flat + i * largeur;
    return X_unflated;
}

// moon-penrose
double train_reg(double* X, int nb_points, int dimension,
           double* Y, int nb_classes, 
           double* centres_out, int nb_centres, double gamma,
           double* W_out){
    
    double** X_mat = unflat(X, dimension, nb_points);
    double** Y_mat = unflat(Y, nb_classes, nb_points);
    
    // Matrices de K centres
    double** centres = k_mean(X_mat, nb_points, dimension, nb_centres);
    // Matrices d'influences Exemple x centres
    double** matrice = remplissage_matricielle_centre(X_mat, nb_points, centres, nb_centres, dimension, gamma);
    // Pseudo-inverse, on embrasse Moone et Penrose
    double** mat_inv = pseudo_inverse(matrice, nb_points, nb_centres);
    // Matrice de poid ->
    double** mat_w   = calcul_poids(mat_inv, Y_mat, nb_centres, nb_points, nb_classes);
    
     // -- Out --
    // mat des centres a plat
    for(int i = 0; i < nb_centres; i++)
        for(int d = 0; d < dimension; d++)
            centres_out[i*dimension + d] = centres[i][d];

    // mat de poids a plat
    for(int c = 0; c < nb_classes; c++)
        for(int j = 0; j < nb_centres; j++)
            W_out[c*nb_centres + j] = mat_w[c][j];
    
    // -- MSE --
    double mse = 0.0;
    for(int i = 0; i < nb_points; i++){
        for(int c = 0; c < nb_classes; c++){
            double score = 0.0;
            for(int j = 0; j < nb_centres; j++)
                score += mat_w[c][j] * influence(X_mat[i], centres[j], gamma, dimension);
            double diff = score - Y_mat[i][c];
            mse += diff * diff;
        }
    }
    mse /= (nb_points * nb_classes);
    
    // free matrices
    liberateur_judiciaire(centres, nb_centres);
    liberateur_judiciaire(matrice, nb_points);
    liberateur_judiciaire(mat_inv, nb_centres);
    liberateur_judiciaire(mat_w, nb_classes);
    
    free(X_mat);
    free(Y_mat);
    
    return mse;
}

//
void init_class(double* X, int nb_points, int dimension, int nb_centres, double gamma,
                  double* centres_out, double* mat_out){
    double** X_mat   = unflat(X, dimension, nb_points);
    double** centres = k_mean(X_mat, nb_points, dimension, nb_centres);
    double** matrice = remplissage_matricielle_centre(X_mat, nb_points, centres, nb_centres, dimension, gamma);

    for(int i = 0; i < nb_centres; i++)
        for(int d = 0; d < dimension; d++)
            centres_out[i*dimension + d] = centres[i][d];
    for(int i = 0; i < nb_points; i++)
        for(int j = 0; j < nb_centres; j++)
            mat_out[i*nb_centres + j] = matrice[i][j];

    liberateur_judiciaire(centres, nb_centres);
    liberateur_judiciaire(matrice, nb_points);
    free(X_mat);
}

// Pour class
double train_epoch(double* mat, int nb_points, int nb_centres,
                   double* Y, int nb_classes, double* W, double learning_rate){
    double** MAT   = unflat(mat, nb_centres, nb_points);
    double** Y_mat = unflat(Y, nb_classes, nb_points);
    double** W_mat = unflat(W, nb_centres, nb_classes);

    double loss = 0.0;
    for(int i = 0; i < nb_points; i++){
        for(int c = 0; c < nb_classes; c++){
            double y_pred = 0.0;
            for(int j = 0; j < nb_centres; j++)
                y_pred += W_mat[c][j] * MAT[i][j];
            double erreur = Y_mat[i][c] - y_pred;
            for(int j = 0; j < nb_centres; j++)
                W_mat[c][j] += learning_rate * erreur * MAT[i][j];
            loss += erreur * erreur;
        }
    }
    free(MAT);
    free(Y_mat);
    free(W_mat);
    
    return loss / (nb_points * nb_classes);
}

void predict(double* X_test, int nb_test, int dimension,
             double* centres, int nb_centres,
             double* W, int nb_classes,
             double gamma, double* scores_out){
    double** X_mat       = unflat(X_test, dimension, nb_test);
    double** centres_mat = unflat(centres, dimension, nb_centres);
    double** W_mat       = unflat(W, nb_centres, nb_classes);

    for(int i = 0; i < nb_test; i++){
        for(int classe = 0; classe < nb_classes; classe++){
            double score = 0.0;
            for(int j = 0; j < nb_centres; j++)
                score += W_mat[classe][j] * influence(X_mat[i], centres_mat[j], gamma, dimension);
            scores_out[i*nb_classes + classe] = score;
        }
    }
    
    free(X_mat);
    free(centres_mat);
    free(W_mat);
}

int main(){
    srand((unsigned int)time(NULL));
    
    /*
    double p0[] = {5.0, 2.0, 9.0};
    double p1[] = {2.0, 6.0, 3.0};
    double p2[] = {4.0, 4.0, 5.0};
    double p3[] = {6.0, 1.0, 2.0};
    double p4[] = {3.0, 8.0, 7.0};
    double p5[] = {9.0, 2.0, 1.0};
    double* X[] = {p0, p1, p2, p3, p4, p5};

    // Y one-hot : p0,p1->classe0 ; p2,p3->classe1 ; p4,p5->classe2
    double y0[] = {1, 0, 0};
    double y1[] = {1, 0, 0};
    double y2[] = {0, 1, 0};
    double y3[] = {0, 1, 0};
    double y4[] = {0, 0, 1};
    double y5[] = {0, 0, 1};
    double* Y[] = {y0, y1, y2, y3, y4, y5};

    int nb_point   = 6;    // nombre d'exemples
    int nb_centre  = 3;    // nombre de centroides RBF (issus du k-means)
    int nb_classes = 3;
    int dimension  = 3;
    double gamma   = 0.03;
    */

    return 0;
}

} // fin externe C*