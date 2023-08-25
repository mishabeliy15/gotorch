#include "torch/torch.h"
#include <iostream>


int main() {
    std::cout << "Is Cuda Available: " << torch::cuda::is_available() << std::endl;
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::Tensor tensor = torch::rand({2, 3}, device=device);
    std::cout << tensor << std::endl;
    return 0;
}