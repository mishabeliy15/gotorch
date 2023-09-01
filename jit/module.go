package jit

// #cgo CFLAGS: -I ${SRCDIR}/..
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch -Wl,-rpath ${SRCDIR}/../cgotorch -lcgotorch
// #cgo LDFLAGS: -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"

import (
	"fmt"
	torch "github.com/wangkuiyi/gotorch"
	"time"
	"unsafe"
)

type Module struct {
	M *unsafe.Pointer
}

// LoadJITModule loads a model from a *.pt file
func LoadJITModule(modelPath string, device *torch.Device) *Module {
	var c C.Module
	if device == nil {
		torch.MustNil(unsafe.Pointer(
			C.loadModule(C.CString(modelPath), nil, &c),
		))
	} else {
		torch.MustNil(unsafe.Pointer(
			C.loadModule(C.CString(modelPath), C.Device(device.T), &c),
		))
	}
	SetModuleFinalizer((*unsafe.Pointer)(&c))
	return &Module{M: (*unsafe.Pointer)(&c)}
}

// Forward runs the forward pass of the model
func (m *Module) Forward(input torch.Tensor) *IValue {
	var c C.IValue
	torch.MustNil(unsafe.Pointer(
		C.forwardModule((C.Module)(*m.M), (C.Tensor)(*input.T), &c),
	))
	fmt.Printf("[%v] Begin SetIValueFinalizer\n", time.Now().UTC())
	SetIValueFinalizer((*unsafe.Pointer)(&c))
	fmt.Printf("[%v] End SetIValueFinalizer\n", time.Now().UTC())
	return &IValue{I: (*unsafe.Pointer)(&c)}
}

func (m *Module) To(device torch.Device) {
	torch.MustNil(unsafe.Pointer(C.Module_ToDevice((C.Module)(*m.M), C.Device(device.T))))
}

func (m *Module) Train(train bool) {
	torch.MustNil(unsafe.Pointer(C.Module_Train((C.Module)(*m.M), C.bool(train))))
}

func (m *Module) Eval() {
	torch.MustNil(unsafe.Pointer(C.Module_Eval((C.Module)(*m.M))))
}
