#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double dsigmoid(double x)
{
    return x * (1 - x);
}

double init_weights()
{
    return ((double)rand()) / ((double)RAND_MAX);
}

void shuffle(int *arr, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = arr[j];
            arr[j] = arr[i];
            arr[i] = t;
        }
    }
}

#define numinputs 2
#define numhiddennodes 2
#define numoutput 1
#define numtrainingsets 4

int main(void)
{
    const double learning_rate = 0.1f;
    double hidden_layer[numhiddennodes];
    double output_layer[numoutput];

    double hidden_layerbias[numhiddennodes];
    double outputlayer_bias[numoutput];

    double hiddenweights[numinputs][numhiddennodes];
    double outputweight[numhiddennodes][numoutput];

    double training_input[numtrainingsets][numinputs] = {{0.0f, 0.0f},
                                                         {1.0f, 0.0f},
                                                         {0.0f, 1.0f},
                                                         {1.0f, 1.0f}};

    double training_output[numtrainingsets][numoutput] = {{0.0f},
                                                          {0.0f},
                                                          {0.0f},
                                                          {1.0f}};

    int i;
    int j;
    for (i = 0; i < numinputs; i++)
    {
        for (j = 0; j < numhiddennodes; j++)
        {
            hiddenweights[i][j] = init_weights();
        }
    }

    for (i = 0; i < numhiddennodes; i++)
    {
        hidden_layerbias[i] = init_weights();
        for (j = 0; j < numoutput; j++)
        {
            outputweight[i][j] = init_weights();
        }
    }

    for (i = 0; i < numoutput; i++)
    {
        outputlayer_bias[i] = init_weights();
    }

    int trainingsetorder[] = {0, 1, 2, 3};
    int numofepochs = 100000000000;
    int epoch;
    int x;
    for (epoch = 0; epoch < numofepochs; epoch++)
    {
        shuffle(trainingsetorder, numtrainingsets);
        for (x = 0; x < numtrainingsets; x++)
        {
            int i = trainingsetorder[x];

            for (j = 0; j < numhiddennodes; j++)
            {
                double activation = hidden_layerbias[j];
                for (int k = 0; k < numinputs; k++)
                {
                    activation += training_input[i][k] * hiddenweights[k][j];
                }
                hidden_layer[j] = sigmoid(activation);
            }

            for (j = 0; j < numoutput; j++)
            {
                double activation = outputlayer_bias[j];
                for (int k = 0; k < numhiddennodes; k++)
                {
                    activation += hidden_layer[k] * outputweight[k][j];
                }
                output_layer[j] = sigmoid(activation);
            }

            printf("Input:%g  Expected output:%g   Output: %g  \n",
                   training_input[i][0], training_input[i][1],
                   output_layer[0], training_output[i][0]);

            double del_output[numoutput];
            for (j = 0; j < numoutput; j++)
            {
                double err_output = (training_output[i][j] - output_layer[j]);
                del_output[j] = err_output * dsigmoid(output_layer[j]);
            }

            double del_hidden[numhiddennodes];
            for (j = 0; j < numhiddennodes; j++)
            {
                double err_hidden = 0.0f;
                for (int k = 0; k < numoutput; k++)
                {
                    err_hidden += del_output[k] * outputweight[j][k];
                }
                del_hidden[j] = err_hidden * dsigmoid(hidden_layer[j]);
            }

            for (j = 0; j < numoutput; j++)
            {
                outputlayer_bias[j] += del_output[j] * learning_rate;
                for (int k = 0; k < numhiddennodes; k++)
                {
                    outputweight[k][j] += hidden_layer[k] * del_output[j] * learning_rate;
                }
            }

            for (j = 0; j < numhiddennodes; j++)
            {
                hidden_layerbias[j] += del_hidden[j] * learning_rate;
                for (int k = 0; k < numinputs; k++)
                {
                    hiddenweights[k][j] += training_input[i][k] * del_hidden[j] * learning_rate;
                }
            }
        }
    }

    fputs("Final hidden weights\n [", stdout);
    for (j = 0; j < numhiddennodes; j++)
    {
        fputs("[ ", stdout);
        for (int k = 0; k < numinputs; k++)
        {
            printf("%f", hiddenweights[k][j]);
        }
        fputs("]", stdout);
    }

    fputs("] \nFinal Hidden Biases \n[", stdout);
    for (j = 0; j < numhiddennodes; j++)
    {
        printf("%lf", hidden_layerbias[j]);
    }

    fputs("Final output weights\n [", stdout);
    for (j = 0; j < numoutput; j++)
    {
        fputs("[ ", stdout);
        for (int k = 0; k < numhiddennodes; k++)
        {
            printf("%f", outputweight[k][j]);
        }
        fputs("] \n", stdout);
    }

    fputs("] \nFinal output Biases \n[", stdout);
    for (j = 0; j < numoutput; j++)
    {
        printf("%lf", outputlayer_bias[j]);
    }

    fputs("] \n", stdout);

    return 0;
}
