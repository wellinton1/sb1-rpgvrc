package gpu

import (
	"fmt"
	"github.com/NVIDIA/cuda-sdk/cuda"
)

type Device struct {
	ID       int
	Memory   uint64
	Name     string
	Compute  string
}

func GetDevices() ([]Device, error) {
	if err := cuda.Init(); err != nil {
		return nil, fmt.Errorf("failed to initialize CUDA: %v", err)
	}

	count, err := cuda.DeviceGetCount()
	if err != nil {
		return nil, fmt.Errorf("failed to get device count: %v", err)
	}

	devices := make([]Device, count)
	for i := 0; i < count; i++ {
		dev := &devices[i]
		dev.ID = i
		
		if err := cuda.DeviceSet(i); err != nil {
			return nil, fmt.Errorf("failed to set device %d: %v", i, err)
		}

		var totalMem uint64
		if err := cuda.DeviceGetAttribute(&totalMem, cuda.DevAttrTotalGlobalMem, i); err != nil {
			return nil, fmt.Errorf("failed to get memory for device %d: %v", i, err)
		}
		dev.Memory = totalMem
	}

	return devices, nil
}