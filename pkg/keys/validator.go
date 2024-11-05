package keys

import (
	"crypto/ecdsa"
	"math/big"
	"sync"

	"github.com/decred/dcrd/dcrec/secp256k1/v4"
)

var (
	curve = secp256k1.S256()
)

type Validator struct {
	curveOrder *big.Int
	cache      sync.Pool
}

func NewValidator() *Validator {
	return &Validator{
		curveOrder: curve.N,
		cache: sync.Pool{
			New: func() interface{} {
				return new(big.Int)
			},
		},
	}
}

func (v *Validator) ValidateKey(key []byte) bool {
	if len(key) != 32 {
		return false
	}

	keyInt := v.cache.Get().(*big.Int)
	defer v.cache.Put(keyInt)

	keyInt.SetBytes(key)
	return keyInt.Cmp(big.NewInt(0)) > 0 && keyInt.Cmp(v.curveOrder) < 0
}

func (v *Validator) ValidateKeyBatch(keys []byte, batchSize int) []bool {
	results := make([]bool, batchSize)
	var wg sync.WaitGroup
	
	// Process in parallel chunks
	chunkSize := 256
	for i := 0; i < batchSize; i += chunkSize {
		end := i + chunkSize
		if end > batchSize {
			end = batchSize
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				key := keys[j*32 : (j+1)*32]
				results[j] = v.ValidateKey(key)
			}
		}(i, end)
	}

	wg.Wait()
	return results
}