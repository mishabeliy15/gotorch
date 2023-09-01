package main

import (
	"fmt"
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/jit"
	"time"
)

func testModel() {
	cuda := torch.IsCUDAAvailable()
	fmt.Printf("CUDA Available: %v\n", cuda)
	var device torch.Device
	if cuda {
		device = torch.NewDevice("cuda")
	} else {
		device = torch.NewDevice("cpu")
	}
	fmt.Printf("[%v] Loading model\n", time.Now().UTC())
	model := jit.LoadJITModule("model.pt", &device)
	fmt.Printf("[%v] Change to eval mode\n", time.Now().UTC())
	model.Eval()
	fmt.Printf("[%v] Creatting tensor\n", time.Now().UTC())
	inputTensor := torch.RandN([]int64{1, 1, 240, 320}, false).To(device)
	fmt.Printf("[%v] Input tensor created successfully! Warm upping...\n", time.Now().UTC())
	model.Forward(inputTensor)
	fmt.Printf("[%v] Warm upped!\n", time.Now().UTC())
	sumDuration := 0.0
	count := 20
	for i := 0; i < count; i++ {
		startTime := time.Now()
		res := model.Forward(inputTensor)
		duration := time.Now().Sub(startTime).Seconds()
		fmt.Printf("[%v] Forward pass time %d: %.5f\n", time.Now().UTC(), i+1, duration)
		sumDuration += duration
		res.Free()
		fmt.Printf("[%v] Freed IValue %d\n", time.Now().UTC(), i+1)
	}
	fmt.Printf("Average time: %.5f\n", sumDuration/float64(count))
}

func main() {
	testModel()
}
