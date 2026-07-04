#ifndef READ_CSV_H
#define READ_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_ROWS 71000
#define NUM_ORIG 21
#define NUM_ENGINEERED 14
#define NUM_FEATURES (NUM_ORIG + NUM_ENGINEERED)
#define TRAIN_SPLIT 0.8

typedef struct {
    float features[NUM_FEATURES];
    int label;
} DataPoint;

// colunas originais do csv (a coluna 0 e o label diabetes):
// HighBP HighChol CholCheck BMI Smoker Stroke HeartDiseaseorAttack
// PhysActivity Fruits Veggies HvyAlcoholConsump AnyHealthcare NoDocbcCost
// GenHlth MentHlth PhysHlth DiffWalk Sex Age Education Income

// cria as features novas a partir das originais (indices 21 ate 34)
void add_features(DataPoint *dp) {
    float *f = dp->features;

    float bp = f[0];
    float chol = f[1];
    float bmi = f[3];
    float stroke = f[5];
    float heart = f[6];
    float phys = f[7];
    float gen = f[13];
    float ment = f[14];
    float physh = f[15];
    float diff = f[16];
    float age = f[18];
    float income = f[20];

    f[21] = bp + chol + diff;                       // riskscore (a que deu melhor gini)
    f[22] = bp + chol + (age >= 9 ? 1 : 0);         // risco metabolico
    f[23] = stroke + heart;                          // ja teve avc/infarto
    f[24] = (physh + ment) / 60.0f;                  // dias ruins no mes normalizado

    f[25] = (bmi >= 25) ? 1 : 0;                     // sobrepeso
    f[26] = (bmi >= 30) ? 1 : 0;                     // obesidade
    f[27] = (bmi >= 35) ? 1 : 0;
    f[28] = (bmi >= 40) ? 1 : 0;

    f[29] = (gen >= 4) ? 1 : 0;                      // saude ruim
    f[30] = (physh >= 15) ? 1 : 0;
    f[31] = (age >= 10) ? 1 : 0;                     // idoso

    f[32] = (bmi >= 30) ? (bp + chol) : 0;           // obeso com pressao/colesterol
    f[33] = (phys < 0.5f && bmi >= 27) ? 1 : 0;      // sedentario com sobrepeso
    f[34] = (income <= 3 && gen >= 3) ? 1 : 0;       // renda baixa + saude ruim
}

int read_csv(const char *filename, DataPoint data[], int *num_samples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("nao consegui abrir o arquivo %s\n", filename);
        return 1;
    }

    char line[4096];
    int row = 0;

    fgets(line, sizeof(line), file); // pula o cabecalho

    while (fgets(line, sizeof(line), file) != NULL && row < MAX_ROWS) {
        line[strcspn(line, "\r\n")] = '\0';
        if (strlen(line) == 0) continue;

        char *tok = strtok(line, ",");
        data[row].label = (int)atof(tok);

        int feat = 0;
        while ((tok = strtok(NULL, ",")) != NULL && feat < NUM_ORIG) {
            data[row].features[feat] = atof(tok);
            feat++;
        }

        if (feat == NUM_ORIG) {
            add_features(&data[row]);
            row++;
        }
    }

    fclose(file);
    *num_samples = row;
    return 0;
}

// embaralha o vetor (fisher-yates)
void shuffle(DataPoint data[], int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        DataPoint tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

void split_data(DataPoint data[], int n, DataPoint train[], DataPoint test[],
                int *train_size, int *test_size) {
    shuffle(data, n);
    int corte = (int)(n * TRAIN_SPLIT);

    for (int i = 0; i < corte; i++)
        train[i] = data[i];
    for (int i = 0; i < n - corte; i++)
        test[i] = data[corte + i];

    *train_size = corte;
    *test_size = n - corte;
}

#endif
