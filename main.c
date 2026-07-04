#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "train.h"

// projeto de random forest pra prever diabetes
// dataset BRFSS 2015, 80% treino e 20% teste

DataPoint *dados;
DataPoint *treino;
DataPoint *teste;

int main() {
    srand(time(NULL));

    printf("Random Forest - Diabetes (BRFSS 2015)\n");
    printf("%d arvores, profundidade %d, %d features por split\n\n",
           N_TREES, MAX_DEPTH, N_FEAT_SPLIT);

    dados  = malloc(MAX_ROWS * sizeof(DataPoint));
    treino = malloc(MAX_ROWS * sizeof(DataPoint));
    teste  = malloc(MAX_ROWS * sizeof(DataPoint));

    int total = 0;
    if (read_csv("data.csv", dados, &total) != 0)
        return 1;
    printf("li %d amostras, %d features cada\n", total, NUM_FEATURES);

    int n_tr, n_te;
    split_data(dados, total, treino, teste, &n_tr, &n_te);
    printf("treino: %d   teste: %d\n\n", n_tr, n_te);
    free(dados);

    printf("treinando...\n");
    clock_t t0 = clock();
    Floresta *rf = treina(treino, n_tr);
    double tempo = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("pronto em %.0f segundos\n\n", tempo);

    // acha o melhor limiar de voto usando o treino
    float *sc_tr = votos(rf, treino, n_tr);
    float lim = melhor_limiar_voto(sc_tr, treino, n_tr);

    int *pred_tr = malloc(n_tr * sizeof(int));
    for (int i = 0; i < n_tr; i++)
        pred_tr[i] = (sc_tr[i] >= lim) ? 1 : 0;
    avalia(pred_tr, treino, n_tr, "TREINO");
    free(sc_tr);
    free(pred_tr);

    // aplica o mesmo limiar no teste
    float *sc_te = votos(rf, teste, n_te);
    int *pred_te = malloc(n_te * sizeof(int));
    for (int i = 0; i < n_te; i++)
        pred_te[i] = (sc_te[i] >= lim) ? 1 : 0;
    avalia(pred_te, teste, n_te, "TESTE");
    free(sc_te);
    free(pred_te);

    libera_floresta(rf);
    free(treino);
    free(teste);
    return 0;
}
