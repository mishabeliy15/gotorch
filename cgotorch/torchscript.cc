#include "torchscript.h"
#include <iostream>


const char* loadModule(const char *modelPath, Device device, Module *result) {
    try {
        if (device != nullptr) {
            *result = new torch::jit::script::Module(torch::jit::load(modelPath, c10::optional<c10::Device>(*device)));
        } else {
            *result = new torch::jit::script::Module(torch::jit::load(modelPath));
        }
        return nullptr;
      } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char* forwardModule(Module module, Tensor input, IValue *output) {
    try {
        c10::IValue forwarded = module->forward({*input});
        *output = new c10::IValue(forwarded);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

void Module_Close(Module a) { delete a; }

const char* Module_ToDevice(Module module, Device device) {
    try {
        module->to(*device);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char* Module_Train(Module module, bool train) {
    try {
        module->train(train);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char* Module_Eval(Module module) {
    try {
        module->eval();
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}
