#ifndef GPUSORT_H_
#define GPUSORT_H_

#ifdef __cplusplus
extern "C" {
#endif

void runSortingKernel(float *keys, int *values, int n);

#ifdef __cplusplus
}
#endif

#endif /* GPUSORT_H_ */
