#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "train.h"

// classificacao de diabetes com random forest - dados do BRFSS 2015

int main() {
    srand(time(NULL));

    DataPoint *dados  = malloc(MAX_ROWS * sizeof(DataPoint));
    DataPoint *treino = malloc(MAX_ROWS * sizeof(DataPoint));
    DataPoint *teste  = malloc(MAX_ROWS * sizeof(DataPoint));
    if (!dados || !treino || !teste) {
        printf("sem memoria suficiente\n");
        return 1;
    }

    int total = 0;
    if (read_csv("data.csv", dados, &total) != 0)
        return 1;

    int n0, n1;
    conta_classes(dados, total, &n0, &n1);
    printf("Foram lidas %d amostras do dataset.\n", total);
    printf("Destas, %d nao tem diabetes (%.1f%%) e %d tem (%.1f%%).\n\n",
           n0, 100.0 * n0 / total, n1, 100.0 * n1 / total);

    int n_tr, n_te;
    split_data(dados, total, treino, teste, &n_tr, &n_te);
    printf("Separando 80%% para treino e 20%% para teste:\n");
    printf("  treino -> %d amostras\n", n_tr);
    printf("  teste  -> %d amostras\n\n", n_te);
    free(dados);

    printf("Treinando o modelo com %d arvores. Isso pode levar um tempo...\n", N_TREES);
    clock_t t0 = clock();
    Floresta *rf = treina(treino, n_tr);
    int seg = (int)((double)(clock() - t0) / CLOCKS_PER_SEC);
    printf("Treino finalizado (%d segundos).\n\n", seg);

    // define o corte de decisao usando so o conjunto de treino
    float *sc_tr = votos(rf, treino, n_tr);
    float lim = melhor_limiar_voto(sc_tr, treino, n_tr);
    printf("Corte de decisao ajustado em %.2f.\n", lim);

    int *pred_tr = malloc(n_tr * sizeof(int));
    for (int i = 0; i < n_tr; i++)
        pred_tr[i] = (sc_tr[i] >= lim) ? 1 : 0;
    avalia(pred_tr, treino, n_tr, "treino");
    free(sc_tr);
    free(pred_tr);

    float *sc_te = votos(rf, teste, n_te);
    int *pred_te = malloc(n_te * sizeof(int));
    for (int i = 0; i < n_te; i++)
        pred_te[i] = (sc_te[i] >= lim) ? 1 : 0;
    avalia(pred_te, teste, n_te, "teste");
    free(sc_te);
    free(pred_te);

    libera_floresta(rf);
    free(treino);
    free(teste);
    return 0;
}
