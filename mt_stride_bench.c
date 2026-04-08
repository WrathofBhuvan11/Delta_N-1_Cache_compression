#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define ARRAY_SIZE (4 * 1024 * 1024)
#define STRIDE     16
#define ITERATIONS 4
#define NUM_THREADS 4

int *arr;
volatile long results[NUM_THREADS];

void *worker(void *arg) {
    int tid = *(int*)arg;
    int chunk = ARRAY_SIZE / NUM_THREADS;
    int start = tid * chunk;
    int end   = start + chunk;
    volatile long sum = 0;
    for (int iter = 0; iter < ITERATIONS; iter++)
        for (int i = start; i < end; i += STRIDE)
            sum += arr[i];
    results[tid] = sum;
    return NULL;
}

int main() {
    arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++)
        arr[i] = (i / 64) * 100 + (i % 64);

    pthread_t threads[NUM_THREADS];
    int tids[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        tids[t] = t;
        pthread_create(&threads[t], NULL, worker, &tids[t]);
    }
    volatile long total = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
        total += results[t];
    }
    printf("sum=%ld\n", total);
    free(arr);
    return 0;
}
