package jit

// #cgo CFLAGS: -I ${SRCDIR}/..
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch -Wl,-rpath ${SRCDIR}/../cgotorch -lcgotorch
// #cgo LDFLAGS: -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"
import (
	"runtime"
	"unsafe"
)

// SetIValueFinalizer sets a finalizer to the IValue
func SetIValueFinalizer(t *unsafe.Pointer) {
	runtime.SetFinalizer(t, func(ct *unsafe.Pointer) {
		if ct != nil {
			C.IValue_Close(C.IValue(*ct))
		}
	})
}
