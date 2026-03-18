#include <stdio.h>
#include <stdlib.h>
#include <math.h>




typedef struct
{
    // Couches
    double inputLayers[2];
    double hiddenLayers[2];
    double outputLayers;

    // Poids
    double w_input_hidden[2][2];
    double w_hidden_output[2];

    // Biais
    double b_hiddenLayers[2];
    double b_outputLayers;

} Mlp;

void init_mlp(Mlp *mlp)
{
    mlp->w_input_hidden[0][0] =  0.5;    //
    mlp->w_input_hidden[0][1] = -0.3;   //  |.5,-.3|
    mlp->w_input_hidden[1][0] =  0.8;  //   |.8, .2|
    mlp->w_input_hidden[1][1] =  0.2; //

    mlp->w_hidden_output[0] =  0.4;
    mlp->w_hidden_output[1] = -0.7;

    mlp->b_hiddenLayers[0] = 0.1;
    mlp->b_hiddenLayers[1] = 0.1;
    mlp->b_outputLayers   = 0.1;
}

// Fonction d'activation -> Sigmoid
double sigmoid(double x)
{
    return 1/(1 + exp(-x));
}

void propagation_avant(Mlp *mlp, double x1, double x2)
{
    mlp->inputLayers[0]=x1;
    mlp->inputLayers[0]=x2;

    //nb de couches caché
    for (int i = 0; i < 2; i++)
    {
        double z = 0;

        for (int j = 0; j < 2; j++)
        {
            // j = neurone de provenant | i = neurone de destination
            z += mlp->inputLayers[j]*mlp->w_input_hidden[j][i];
        }

        z += mlp->b_hiddenLayers[i];
        mlp->hiddenLayers[i] = sigmoid(z);
    }

    double z_output = 0;
    for (int i = 0; i < 2; i++)
    {
        z_output += mlp->hiddenLayers[i] * mlp->w_hidden_output[i];
    }

    z_output += mlp->b_outputLayers;
    mlp->outputLayers = sigmoid(z_output);
}

int main(void)
{
    Mlp mlp;

    init_mlp(&mlp);
    propagation_avant(&mlp, 1.0, 0.0);

    printf("H1    : %f\n", mlp.hiddenLayers[0]);
    printf("H2    : %f\n", mlp.hiddenLayers[1]);
    printf("Out   : %f\n", mlp.outputLayers);

    return 0;
}