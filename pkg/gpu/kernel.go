package gpu

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/NVIDIA/cuda-sdk/cuda"
)

const (
	BlockSize  = 512      // Increased block size for better occupancy
	NumBlocks  = 2048     // More blocks for higher throughput
	BatchSize  = BlockSize * NumBlocks
	KeySize    = 32
	NumStreams = 4        // Multiple CUDA streams for overlapping operations
)

type Kernel struct {
	device     *Device
	streams    [NumStreams]cuda.Stream
	memPools   [NumStreams]unsafe.Pointer
	params     cuda.LaunchParams
	mu         sync.Mutex
}

func NewKernel(device *Device) (*Kernel, error) {
	if err := cuda.DeviceSet(device.ID); err != nil {
		return nil, fmt.Errorf("failed to set device: %v", err)
	}

	k := &Kernel{
		device: device,
		params: cuda.LaunchParams{
			GridDimX:  NumBlocks,
			GridDimY:  1,
			GridDimZ:  1,
			BlockDimX: BlockSize,
			BlockDimY: 1,
			BlockDimZ: 1,
		},
	}

	// Initialize CUDA streams and memory pools
	for i := 0; i < NumStreams; i++ {
		stream, err := cuda.StreamCreate()
		if err != nil {
			k.Close()
			return nil, fmt.Errorf("failed to create stream %d: %v", i, err)
		}
		k.streams[i] = stream

		pool, err := cuda.MallocAsync(BatchSize*KeySize)
		if err != nil {
			k.Close()
			return nil, fmt.Errorf("failed to allocate device memory for stream %d: %v", i, err)
		}
		k.memPools[i] = pool
	}

	return k, nil
}

func (k *Kernel) GenerateKeys() ([][]byte, error) {
	k.mu.Lock()
	defer k.mu.Unlock()

	results := make([][]byte, NumStreams)
	var wg sync.WaitGroup

	for i := 0; i < NumStreams; i++ {
		wg.Add(1)
		go func(streamIndex int) {
			defer wg.Done()

			// Launch kernel asynchronously
			stream := k.streams[streamIndex]
			if err := cuda.LaunchKernelAsync(generateKeys, k.params, k.memPools[streamIndex], stream); err != nil {
				results[streamIndex] = nil
				return
			}

			// Allocate host memory and copy results
			keys := make([]byte, BatchSize*KeySize)
			if err := cuda.MemcpyDtoHAsync(keys, k.memPools[streamIndex], stream); err != nil {
				results[streamIndex] = nil
				return
			}

			cuda.StreamSynchronize(stream)
			results[streamIndex] = keys
		}(i)
	}

	wg.Wait()

	// Combine valid results
	var allKeys []byte
	for _, keys := range results {
		if keys != nil {
			allKeys = append(allKeys, keys...)
		}
	}

	return allKeys, nil
}

func (k *Kernel) Close() error {
	for i := 0; i < NumStreams; i++ {
		if k.memPools[i] != nil {
			cuda.Free(k.memPools[i])
		}
		if k.streams[i] != nil {
			cuda.StreamDestroy(k.streams[i])
		}
	}
	return nil
}