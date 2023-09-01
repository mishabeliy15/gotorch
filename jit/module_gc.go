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

// SetModuleFinalizer sets a finalizer to the Module
func SetModuleFinalizer(t *unsafe.Pointer) {
	runtime.SetFinalizer(t, func(ct *unsafe.Pointer) {
		if ct != nil {
			C.Module_Close(C.Module(*ct))
		}
	})
}
