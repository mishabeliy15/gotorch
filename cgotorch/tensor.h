/* Copyright 2020, GoTorch Authors */
#pragma once

#include "torch.h"

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations, torch
////////////////////////////////////////////////////////////////////////////////

const char *Tensor_Detach(Tensor a, Tensor *result);
const char *Tensor_String(Tensor a);
void Tensor_Print(Tensor a);
void Tensor_Close(Tensor a);
void FreeString(const char *s);
const char *Tensor_Save(Tensor tensor, const char *path);
const char *Tensor_Load(const char *path, Tensor *result);
const char *Tensor_Dim(Tensor tensor, int64_t *dim);
const char *Tensor_Shape(Tensor tensor, int64_t *dims);
const char *Tensor_Dtype(Tensor tensor, int8_t *dtype);
const char *Tensor_SetData(Tensor self, Tensor new_data);
const char *Tensor_FromBlob(void *data, int8_t dtype, int64_t *sizes_data,
                            int64_t sizes_data_len, Tensor *result);
const char *Tensor_To(Tensor input, Device device, int8_t dtype,
                      Tensor *output);
const char *Tensor_CastTo(Tensor input, int8_t dtype, Tensor *output);
const char *Tensor_CopyTo(Tensor input, Device device, Tensor *output);
const char *Tensor_PinMemory(Tensor input, Tensor *output);
const char *Tensor_CUDA(Tensor input, Device device, int8_t non_blocking,
                        Tensor *output);
const char *Tensor_Reshape(Tensor input, int64_t *shape, int64_t shape_len,
                           Tensor *result);
const char *Tensor_Split(Tensor input, int64_t split_size, int64_t dim,
                         Tensor *results, int64_t *results_len);
const char *Tensor_Slice(Tensor input, int64_t dim, int64_t start, int64_t end,
                         int64_t step, Tensor *result);
const char *Tensor_Norm(Tensor input, int64_t p, int64_t dim, Tensor *result);
const char *Tensor_Unsqueeze(Tensor input, int64_t dim, Tensor *result);
const char *Tensor_GeScalar(Tensor input, float other, Tensor *result);
const char *Tensor_NonZero(Tensor input, Tensor *result);
const char *Tensor_Zeros(int8_t dtype, int64_t *sizes_data,
                         int64_t sizes_data_len, Tensor *result);
const char *Tensor_IndexPut(Tensor input, int64_t index, Tensor source);
const char *Tensor_IndexByTensors(Tensor input, Tensor *indexes, int64_t index_len, Tensor *result);
const char *Tensor_Device(Tensor input, Device *device);
////////////////////////////////////////////////////////////////////////////////
// Backward, Gradient
////////////////////////////////////////////////////////////////////////////////

void Tensor_Backward(Tensor a);
Tensor Tensor_Grad(Tensor a);

////////////////////////////////////////////////////////////////////////////////
// Get elements
////////////////////////////////////////////////////////////////////////////////

const char *Item(Tensor a, float *result);
const char *ItemInt64(Tensor a, int64_t *result);
const char *ItemFloat64(Tensor a, double *result);

const char *Tensor_Index(Tensor a, int64_t *index, int64_t index_len,
                         Tensor *result);
const char *Tensor_ToArray(Tensor input, void *result);
const char *Tensor_Select(Tensor input, int64_t dim, int64_t index,
                          Tensor *result);

#ifdef __cplusplus
}
#endif
