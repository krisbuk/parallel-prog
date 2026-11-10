#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Использование: %s <размер> <выходной_файл> <сид>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    const char* filename = argv[2];
    unsigned seed = (unsigned)atoi(argv[3]);

    srand(seed);
    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Ошибка открытия файла");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%.4f ", (float)rand() / RAND_MAX * 10.0);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Матрица %dx%d сохранена в %s\n", n, n, filename);
    return 0;
}