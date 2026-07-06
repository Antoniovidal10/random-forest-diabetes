#ifndef TRAIN_H
#define TRAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "read_csv.h"

#define N_TREES 300
#define MAX_DEPTH 12
#define MIN_SAMPLES 30
#define N_FEAT_SPLIT 6
#define PESO_POSITIVO 3   // no bootstrap, diabeticos sao sorteados com esse peso
#define MAX_THRESH 20

typedef struct No {
    int feature;
    double limiar;
    int classe;
    struct No *esq, *dir;
    int folha;
} No;

typedef struct {
    No **arvores;
    int n;
} Floresta;

int compara(const void *a, const void *b) {
    double x = *(double*)a, y = *(double*)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

double gini(DataPoint *s, int n) {
    if (n == 0) return 0;
    int uns = 0;
    for (int i = 0; i < n; i++)
        uns += s[i].label;
    double p = (double)uns / n;
    return 1.0 - p*p - (1-p)*(1-p);
}

int classe_maioria(DataPoint *s, int n) {
    int uns = 0;
    for (int i = 0; i < n; i++)
        uns += s[i].label;
    if (uns * 2 >= n) return 1;
    return 0;
}

// coloca os menores que o limiar na esquerda, retorna quantos ficaram la
int separa(DataPoint *v, int n, int feat, double lim) {
    int e = 0;
    for (int i = 0; i < n; i++) {
        if (v[i].features[feat] < lim) {
            DataPoint tmp = v[e];
            v[e] = v[i];
            v[i] = tmp;
            e++;
        }
    }
    return e;
}

// pega os limiares possiveis (meio termo entre os valores distintos da feature)
int pega_limiares(DataPoint *s, int n, int feat, double *out, int max) {
    double *vals = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        vals[i] = s[i].features[feat];
    qsort(vals, n, sizeof(double), compara);

    int u = 0;
    for (int i = 0; i < n; i++)
        if (u == 0 || vals[i] != vals[u-1])
            vals[u++] = vals[i];

    int c = 0;
    if (u <= 1) {
        free(vals);
        return 0;
    }
    if (u == 2) {
        out[c++] = (vals[0] + vals[1]) / 2.0;
    } else {
        int passo = (u - 1 <= max) ? 1 : (u / max);
        for (int i = 0; i < u - 1 && c < max; i += passo)
            out[c++] = (vals[i] + vals[i+1]) / 2.0;
    }
    free(vals);
    return c;
}

void sorteia_features(int *idx) {
    int pool[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++)
        pool[i] = i;
    for (int i = NUM_FEATURES - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = pool[i]; pool[i] = pool[j]; pool[j] = t;
    }
    for (int i = 0; i < N_FEAT_SPLIT; i++)
        idx[i] = pool[i];
}

void melhor_corte(DataPoint *s, int n, int *bfeat, double *blim, double *bgini) {
    *bgini = 2.0;
    *bfeat = -1;
    *blim = 0;

    int feats[N_FEAT_SPLIT];
    sorteia_features(feats);

    DataPoint *buf = malloc(n * sizeof(DataPoint));
    double lims[MAX_THRESH + 5];

    for (int fi = 0; fi < N_FEAT_SPLIT; fi++) {
        int feat = feats[fi];
        int nl = pega_limiares(s, n, feat, lims, MAX_THRESH);

        for (int t = 0; t < nl; t++) {
            memcpy(buf, s, n * sizeof(DataPoint));
            int esq = separa(buf, n, feat, lims[t]);
            int dir = n - esq;
            if (esq == 0 || dir == 0) continue;

            double g = (esq * gini(buf, esq) + dir * gini(buf + esq, dir)) / n;
            if (g < *bgini) {
                *bgini = g;
                *bfeat = feat;
                *blim = lims[t];
            }
        }
    }
    free(buf);
}

No *cria_arvore(DataPoint *s, int n, int prof) {
    No *no = calloc(1, sizeof(No));

    double g = gini(s, n);
    if (prof >= MAX_DEPTH || n < MIN_SAMPLES || g < 1e-7) {
        no->folha = 1;
        no->classe = classe_maioria(s, n);
        return no;
    }

    int bfeat;
    double blim, bgini;
    melhor_corte(s, n, &bfeat, &blim, &bgini);

    if (bfeat == -1 || bgini >= g - 1e-7) {
        no->folha = 1;
        no->classe = classe_maioria(s, n);
        return no;
    }

    DataPoint *buf = malloc(n * sizeof(DataPoint));
    memcpy(buf, s, n * sizeof(DataPoint));
    int esq = separa(buf, n, bfeat, blim);

    no->feature = bfeat;
    no->limiar = blim;
    no->esq = cria_arvore(buf, esq, prof + 1);
    no->dir = cria_arvore(buf + esq, n - esq, prof + 1);

    free(buf);
    return no;
}

void libera_arvore(No *no) {
    if (no == NULL) return;
    libera_arvore(no->esq);
    libera_arvore(no->dir);
    free(no);
}

void libera_floresta(Floresta *f) {
    for (int i = 0; i < f->n; i++)
        libera_arvore(f->arvores[i]);
    free(f->arvores);
    free(f);
}

Floresta *treina(DataPoint train[], int n) {
    Floresta *f = malloc(sizeof(Floresta));
    f->n = N_TREES;
    f->arvores = malloc(N_TREES * sizeof(No*));

    // monta uma "urna" onde cada diabetico entra PESO_POSITIVO vezes.
    // assim o bootstrap sorteia a classe minoritaria com mais frequencia
    // e o modelo para de ignorar os diabeticos.
    int tam_urna = 0;
    for (int i = 0; i < n; i++)
        tam_urna += (train[i].label == 1) ? PESO_POSITIVO : 1;

    int *urna = malloc(tam_urna * sizeof(int));
    int k = 0;
    for (int i = 0; i < n; i++) {
        int rep = (train[i].label == 1) ? PESO_POSITIVO : 1;
        for (int r = 0; r < rep; r++)
            urna[k++] = i;
    }

    DataPoint *amostra = malloc(n * sizeof(DataPoint));

    for (int t = 0; t < N_TREES; t++) {
        for (int i = 0; i < n; i++)
            amostra[i] = train[urna[rand_grande() % tam_urna]];

        f->arvores[t] = cria_arvore(amostra, n, 0);

        if ((t + 1) % 50 == 0 || t == N_TREES - 1)
            printf("  arvore %d de %d\n", t + 1, N_TREES);
        fflush(stdout);
    }

    free(amostra);
    free(urna);
    return f;
}

int prediz_arvore(No *no, DataPoint *s) {
    while (!no->folha) {
        if (s->features[no->feature] < no->limiar)
            no = no->esq;
        else
            no = no->dir;
    }
    return no->classe;
}

// fracao de arvores que votaram 1 pra cada amostra
float *votos(Floresta *f, DataPoint *s, int n) {
    float *sc = calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        int v = 0;
        for (int t = 0; t < f->n; t++)
            v += prediz_arvore(f->arvores[t], &s[i]);
        sc[i] = (float)v / f->n;
    }
    return sc;
}

// procura o limiar de decisao que da mais acerto no treino
// procura o corte de decisao que da o melhor f1 no treino.
// uso f1 (e nao acuracia) pra funcionar bem tanto no balanceado
// quanto no desbalanceado, onde a acuracia sozinha engana.
float melhor_limiar_voto(float *sc, DataPoint *train, int n) {
    float melhor = 0.5, melhor_f1 = 0;
    for (int t = 20; t <= 65; t++) {
        float lim = t / 100.0f;
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < n; i++) {
            int pred = (sc[i] >= lim) ? 1 : 0;
            if (pred == 1 && train[i].label == 1) tp++;
            else if (pred == 1 && train[i].label == 0) fp++;
            else if (pred == 0 && train[i].label == 1) fn++;
        }
        float pre = (tp + fp > 0) ? (float)tp / (tp + fp) : 0;
        float rec = (tp + fn > 0) ? (float)tp / (tp + fn) : 0;
        float f1 = (pre + rec > 0) ? 2 * pre * rec / (pre + rec) : 0;
        if (f1 > melhor_f1) {
            melhor_f1 = f1;
            melhor = lim;
        }
    }
    return melhor;
}

void avalia(int *pred, DataPoint *dados, int n, const char *nome) {
    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (int i = 0; i < n; i++) {
        int p = pred[i];
        int y = dados[i].label;
        if (p == 1 && y == 1) tp++;
        else if (p == 1 && y == 0) fp++;
        else if (p == 0 && y == 0) tn++;
        else fn++;
    }

    float err = (float)(fp + fn) / (tp + fp + tn + fn);
    float acc = 1 - err;
    float pre = (tp + fp > 0) ? (float)tp / (tp + fp) : 0;
    float rec = (tp + fn > 0) ? (float)tp / (tp + fn) : 0;
    float f1 = (pre + rec > 0) ? 2 * pre * rec / (pre + rec) : 0;

    printf("\nResultados no conjunto de %s:\n", nome);
    printf("  acertou %d de %d amostras\n", tp + tn, n);
    printf("  dos %d diabeticos reais, o modelo encontrou %d\n", tp + fn, tp);
    printf("  deu %d alarmes falsos (disse que tinha, mas nao tinha)\n\n", fp);
    printf("  acuracia .. %.2f%%\n", acc * 100);
    printf("  erro ...... %.2f%%\n", err * 100);
    printf("  precisao .. %.2f%%\n", pre * 100);
    printf("  recall .... %.2f%%\n", rec * 100);
    printf("  f1 ........ %.2f%%\n", f1 * 100);
}

#endif
