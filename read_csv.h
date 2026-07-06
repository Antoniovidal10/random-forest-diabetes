#ifndef READ_CSV_H
#define READ_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_ROWS 260000
#define NUM_ORIG 21
#define NUM_EXTRA 6
#define NUM_FEATURES (NUM_ORIG + NUM_EXTRA)
#define TRAIN_SPLIT 0.8

typedef struct {
    float features[NUM_FEATURES];
    int label;
} DataPoint;

// colunas: 0=HighBP 1=HighChol 2=CholCheck 3=BMI 4=Smoker 5=Stroke
// 6=HeartDiseaseorAttack 7=PhysActivity 8=Fruits 9=Veggies 10=HvyAlcoholConsump
// 11=AnyHealthcare 12=NoDocbcCost 13=GenHlth 14=MentHlth 15=PhysHlth
// 16=DiffWalk 17=Sex 18=Age 19=Education 20=Income

// algumas features extras que testei e melhoraram um pouco o resultado
void add_features(DataPoint *dp) {
    float *f = dp->features;

    f[21] = f[0] + f[1] + f[16];         // pressao + colesterol + dificuldade de andar
    f[22] = f[5] + f[6];                 // avc ou infarto
    f[23] = (f[3] >= 30) ? 1 : 0;        // obesidade
    f[24] = (f[13] >= 4) ? 1 : 0;        // saude geral ruim
    f[25] = (f[18] >= 10) ? 1 : 0;       // idoso
    f[26] = f[15] + f[14];              // dias ruins fisico + mental
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
        int y = (int)atof(tok);
        data[row].label = (y >= 1) ? 1 : 0;   // junta pre-diabetes e diabetes

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

void conta_classes(DataPoint data[], int n, int *n0, int *n1) {
    *n0 = 0;
    *n1 = 0;
    for (int i = 0; i < n; i++) {
        if (data[i].label == 0) (*n0)++;
        else (*n1)++;
    }
}

// no windows o RAND_MAX e 32767, junto dois rand pra alcancar indices maiores
int rand_grande() {
    return ((rand() << 15) | rand()) & 0x3FFFFFFF;
}

void shuffle(DataPoint data[], int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand_grande() % (i + 1);
        DataPoint tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

// divide 80/20 mantendo a mesma proporcao de cada classe nos dois lados
void split_data(DataPoint data[], int n, DataPoint train[], DataPoint test[],
                int *train_size, int *test_size) {
    shuffle(data, n);

    // separa as duas classes
    DataPoint *c0 = malloc(n * sizeof(DataPoint));
    DataPoint *c1 = malloc(n * sizeof(DataPoint));
    int n0 = 0, n1 = 0;
    for (int i = 0; i < n; i++) {
        if (data[i].label == 0) c0[n0++] = data[i];
        else c1[n1++] = data[i];
    }

    int corte0 = (int)(n0 * TRAIN_SPLIT);
    int corte1 = (int)(n1 * TRAIN_SPLIT);

    int tr = 0, te = 0;
    for (int i = 0; i < n0; i++) {
        if (i < corte0) train[tr++] = c0[i];
        else test[te++] = c0[i];
    }
    for (int i = 0; i < n1; i++) {
        if (i < corte1) train[tr++] = c1[i];
        else test[te++] = c1[i];
    }

    // embaralha de novo pra nao ficar tudo de uma classe junto
    shuffle(train, tr);
    shuffle(test, te);

    *train_size = tr;
    *test_size = te;
    free(c0);
    free(c1);
}

#endif
