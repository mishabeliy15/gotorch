package main

import (
	"fmt"
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/jit"
	"time"
)

func testModel() {
	fmt.Printf("[%v] Loading model\n", time.Now().UTC())
	model := jit.LoadJITModule("model.pt", nil)
	fmt.Printf("[%v] Model loaded successfully!\n", time.Now().UTC())
	inputTensor := torch.RandN([]int64{1, 1, 240, 320}, false)
	fmt.Printf("[%v] Input tensor created successfully!", time.Now().UTC())
	res := model.Forward(inputTensor)
	fmt.Printf("[%v] res is tuple: %v\n", time.Now().UTC(), res.IsTuple())
	tuple := res.ToTuple()
	for i, t := range tuple {
		fmt.Printf("[%v] tuple[%d] is tensor: %v\n", time.Now().UTC(), i, t.IsTensor())
		tensor := t.ToTensor().To(torch.NewDevice("cpu"), torch.Float)
		shapes, sl := tensor.ToFloat32Slice()
		fmt.Printf("[%v] tensor shape: %v\nTensor len: %v\n", time.Now().UTC(), shapes, len(sl))
	}
	var device torch.Device
	cudaAvailable := torch.IsCUDAAvailable()
	fmt.Printf("[%v] CUDA Available: %v\n", time.Now().UTC(), cudaAvailable)
	if cudaAvailable {
		device = torch.NewDevice("cuda")
	} else {
		device = torch.NewDevice("cpu")
	}
	fmt.Printf("[%v] Moving model to device: %v\n", time.Now().UTC(), device)
	model.To(device)
	fmt.Printf("[%v] Moving input tensor to device: %v\n", time.Now().UTC(), device)
	inputTensor = inputTensor.To(device)
	fmt.Printf("[%v] Run forward pass\n", time.Now().UTC())
	res = model.Forward(inputTensor)
	fmt.Printf("[%v] res is tuple: %v\n", time.Now().UTC(), res.IsTuple())
}

func main() {
	testModel()
}
