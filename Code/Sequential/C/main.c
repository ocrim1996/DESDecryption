#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <crypt.h>

#define WORD_SIZE 8 // word size
#define SALT "PC" // 2 characters salt is needed to use DES in crypt() func
#define DICTIONARY "dict_1500000.txt"

float execute(const char * password, int nAverage) {
    printf("[*] target: %s\n", password);
    printf("[*] salt: %s\n", SALT);

    char* encrypted = strdup(crypt(password, SALT));
    printf("[*] encrypted string: %s\n", encrypted);

    printf("[*] Measuring average time...\n");

    float executionTimes[nAverage];
    for (int i=0; i<nAverage; i++) {

        FILE* dictionary;
        if((dictionary = fopen(DICTIONARY, "r")) == NULL) {
            fprintf(stderr, "[?] error: dictionary not found\n");
            exit (EXIT_FAILURE);
        }

        bool found = false;
        size_t len = 0;
        char* current_word = (char*)malloc(WORD_SIZE * sizeof(char));
        char* current_word_encrypted = NULL;

        // Start clock
        clock_t start = clock();

        while ((getline(&current_word, &len, dictionary)) != -1) {
            //printf("[*] checking word: %s", current_word);

            current_word_encrypted = strdup(crypt(current_word, SALT));
            //printf("[*] encrypted word: %s\n", current_word_encrypted);

            if (strcmp(current_word_encrypted, encrypted) == 0) {
                //printf("[✓] PASSWORD FOUND IN DICTIONARY: %s", current_word);

                clock_t end = clock();

                float elapsedTime = (float)(end - start)/CLOCKS_PER_SEC;
                //printf("[i] CPU time: %.5f seconds\n", elapsedTime);

                executionTimes[i] = elapsedTime;

                found = true;
                break;
            }
        }

        if (!found) {
            printf("[?] password not found in dictionary!\n");
            executionTimes[i] = 0;
        }

        fclose(dictionary);
        free(current_word);
        free(current_word_encrypted);
    }

    printf("[*] Cleaning up...\n");

    free(encrypted);

    // Get average
    float avg = 0;
    float sum = 0;
    for (int i=0; i<nAverage; i++) {
        sum += executionTimes[i];
    }
    avg = (float)sum/nAverage;

    printf("[*] Done! Average time: %.5f", avg);
    return avg;
}

int main() {

    float times[10];
    const char *passwords[10] = {"sirpizza", "3kingdom", "tyleisha", "marumari", "giacomix", "dbcookie2", "Yessssss", "Mypaypa12", "6Melissa", "1Mazzola"};

    for (int i=0; i<10; i++) {
        times[i] = execute(passwords[i], 5);
        printf("\n\n");
    }

    // Print execution times
    printf("\n\n[*] Times: ");
    for (int z=0; z<10; z++) {
        printf(z==9 ? "%f" : "%f, ", times[z]);
    }

    return 0;
}
