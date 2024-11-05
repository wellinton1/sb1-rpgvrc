package main

import (
	"log"
	"runtime"
	"sync"
	"time"

	"btcgo/pkg/gpu"
	"btcgo/pkg/keys"
)

func main() {
	// Set maximum CPU threads for parallel processing
	runtime.GOMAXPROCS(runtime.NumCPU())

	devices, err := gpu.GetDevices()
	if err != nil {
		log.Fatalf("Failed to get GPU devices: %v", err)
	}

	validator := keys.NewValidator()
	var wg sync.WaitGroup

	// Start processing on each GPU
	for _, device := range devices {
		wg.Add(1)
		go func(dev gpu.Device) {
			defer wg.Done()
			processDevice(dev, validator)
		}(device)
	}

	wg.Wait()
}

func processDevice(device gpu.Device, validator *keys.Validator) {
	kernel, err := gpu.NewKernel(&device)
	if err != nil {
		log.Printf("Failed to initialize kernel for device %d: %v", device.ID, err)
		return
	}
	defer kernel.Close()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	var keyCount uint64
	start := time.Now()

	for {
		select {
		case <-ticker.C:
			elapsed := time.Since(start).Seconds()
			rate := float64(keyCount) / elapsed
			log.Printf("Device %d: %.2f keys/sec", device.ID, rate)
		default:
			keys, err := kernel.GenerateKeys()
			if err != nil {
				log.Printf("Failed to generate keys on device %d: %v", device.ID, err)
				continue
			}

			batchSize := len(keys) / 32
			validKeys := validator.ValidateKeyBatch(keys, batchSize)
			keyCount += uint64(batchSize)

			processValidKeys(keys, validKeys)
		}
	}
}

func processValidKeys(keys []byte, validFlags []bool) {
	for i, valid := range validFlags {
		if valid {
			key := keys[i*32 : (i+1)*32]
			// Process valid key (implementation specific)
			_ = key
		}
	}
}