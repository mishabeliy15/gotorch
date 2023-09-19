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

type IValue struct {
	I *unsafe.Pointer
}

// IsTuple returns true if the IValue is a tuple
func (i *IValue) IsTuple() bool {
	return bool(C.IValue_isTuple((C.IValue)(*i.I)))
}

// IsTensor returns true if the IValue is a tensor
func (i *IValue) IsTensor() bool {
	return bool(C.IValue_isTensor((C.IValue)(*i.I)))
}

// ToTuple converts the IValue to a tuple (go slice)
func (i *IValue) ToTuple() []*IValue {
	var c *C.IValue
	var cLength C.int
	torch.MustNil(unsafe.Pointer(
		C.IValue_toTuple((C.IValue)(*i.I), &c, &cLength),
	))
	length := int(cLength)
	results := make([]*IValue, length)
	for i := 0; i < length; i++ {
		results[i] = &IValue{
			I: (*unsafe.Pointer)(unsafe.Pointer(uintptr(unsafe.Pointer(c)) + uintptr(i)*unsafe.Sizeof(c))),
		}
	}
	return results
}

// ToTensor converts the IValue to a tensor
func (i *IValue) ToTensor() torch.Tensor {
	var c C.Tensor
	torch.MustNil(unsafe.Pointer(
		C.IValue_toTensor((C.IValue)(*i.I), &c),
	))
	return torch.Tensor{T: (*unsafe.Pointer)(&c)}
}

// Free frees the IValue in C++
func (i *IValue) Free() {
	if i.I != nil {
		C.IValue_Close(C.IValue(*i.I))
		i.I = nil
	}
}
