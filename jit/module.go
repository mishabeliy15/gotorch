package jit

// #cgo CFLAGS: -I ${SRCDIR}/..
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch -Wl,-rpath ${SRCDIR}/../cgotorch -lcgotorch
// #cgo LDFLAGS: -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"

import (
	torch "github.com/wangkuiyi/gotorch"
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
	SetIValueFinalizer((*unsafe.Pointer)(&c))
	return &IValue{I: (*unsafe.Pointer)(&c)}
}

// To move the model to a device
func (m *Module) To(device torch.Device) {
	torch.MustNil(unsafe.Pointer(C.Module_ToDevice((C.Module)(*m.M), C.Device(device.T))))
}

// Train sets the model to train mode
func (m *Module) Train(train bool) {
	torch.MustNil(unsafe.Pointer(C.Module_Train((C.Module)(*m.M), C.bool(train))))
}

// Eval sets the model to eval mode
func (m *Module) Eval() {
	torch.MustNil(unsafe.Pointer(C.Module_Eval((C.Module)(*m.M))))
}

// Free frees the model in C++
func (m *Module) Free() {
	if m.M != nil {
		C.Module_Close(C.Module(*m.M))
		m.M = nil
	}
}
