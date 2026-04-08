#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

/*
 * STREAM benchmark - Copy / Scale / Add / Triad kernels
 * Compile: gcc -O2 -static -pthread -o mt16_stream mt16_stream.c
 * Run:     ./mt16_stream
 *
 * N = 256K doubles -> 2MB per array, 6MB total
 * Exceeds 16 × 256KB aggregate L2 -> all traffic hits DRAM / Ruby directory
 */

#define N           (256 * 1024)
#define NTIMES      2
#define SCALAR      3
#define NUM_THREADS 16

/* --- global arrays (heap, malloc'd in main - same as stride_bench) --- */
double *a;
double *b;
double *c;

/* --- per-thread results (same pattern as stride_bench) --- */
volatile double results[NUM_THREADS];

/* --- barrier: syncs all threads between kernels --- */
static pthread_barrier_t bar;

/* --- global thread count (avoids struct, matches stride_bench style) --- */
static int g_nthreads;

void *worker(void *arg)
{
    int tid   = *(int *)arg;
    int chunk = N / g_nthreads;
    int start = tid * chunk;
    int end   = start + chunk;
    int k, i;
    volatile double sum = 0;

    /* Copy: c[i] = a[i] */
    for (k = 0; k < NTIMES; k++) {
        pthread_barrier_wait(&bar);
        for (i = start; i < end; i++)
            c[i] = a[i];
        pthread_barrier_wait(&bar);
    }

    /* Scale: b[i] = SCALAR * c[i] */
    for (k = 0; k < NTIMES; k++) {
        pthread_barrier_wait(&bar);
        for (i = start; i < end; i++)
            b[i] = SCALAR * c[i];
        pthread_barrier_wait(&bar);
    }

    /* Add: c[i] = a[i] + b[i] */
    for (k = 0; k < NTIMES; k++) {
        pthread_barrier_wait(&bar);
        for (i = start; i < end; i++)
            c[i] = a[i] + b[i];
        pthread_barrier_wait(&bar);
    }

    /* Triad: a[i] = b[i] + SCALAR * c[i]  <-- canonical STREAM kernel */
    for (k = 0; k < NTIMES; k++) {
        pthread_barrier_wait(&bar);
        for (i = start; i < end; i++)
            a[i] = b[i] + SCALAR * c[i];
        pthread_barrier_wait(&bar);
    }

    for (i = start; i < end; i++)
        sum += a[i];
    results[tid] = sum;
    return NULL;
}

int main()
{
    g_nthreads = NUM_THREADS;

    a = (double *)malloc(N * sizeof(double));
    b = (double *)malloc(N * sizeof(double));
    c = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    pthread_barrier_init(&bar, NULL, NUM_THREADS);

    pthread_t threads[NUM_THREADS];
    int tids[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        tids[t] = t;
        pthread_create(&threads[t], NULL, worker, &tids[t]);
    }

    volatile double total = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
        total += results[t];
    }

    printf("checksum=%.2f\n", total);

    pthread_barrier_destroy(&bar);
    free(a); free(b); free(c);
    return 0;
}

