#include "torch/torch.h"
#include <iostream>


int main() {
    std::cout << "Is Cuda Available: " << torch::cuda::is_available() << std::endl;
    torch::Tensor tensor = torch::rand({2, 3});
    tensor = tensor.to(torch::kCUDA);
    std::cout << tensor << std::endl;
    return 0;
}