#pragma once

#include "torchdef.h"

#ifdef __cplusplus
#include "torch/script.h"

extern "C" {
typedef torch::jit::script::Module *Module;
#else
typedef void *Module;
#endif

const char* loadModule(const char *modelPath, Device device, Module *result);
const char* forwardModule(Module module, Tensor input, IValue *output);
void Module_Close(Module a);
const char* Module_ToDevice(Module module, Device device);

#ifdef __cplusplus
}
#endif
