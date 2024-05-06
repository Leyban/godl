package activation

import (
	"godl/model"
	"math"
)

// Rectified Linear Unit Function
// Args:
// Z -- Outputs of the Linear Layer
func ReLU(Z [][]float64) (A [][]float64) {
	A = make([][]float64, len(Z))
	for m := range A {
		A[m] = make([]float64, len(Z[m]))
	}

	for m := range A {
		for n := range A[m] {
			A[m][n] = math.Max(0, Z[m][n])
		}
	}

	return A
}

// calculate the g'(z) for ReLU
// dA[n][m]
// dZ[n][m]
// dZ is 0 if Z is <= 0; otherwise it's 1
func ReLUBAckward(dA [][]float64, cache model.ForwardPropCache) (dZ [][]float64) {

	dZ = make([][]float64, len(dA))

	for n := range dA {
		dZ[n] = make([]float64, len(dA[n]))
		for m := range dA[n] {
			if cache.Z[n][m] > 0 {
				dZ[n][m] = dA[n][m]
				// dZ[n][m] = 1 // correct way
			}
		}
	}

	return dZ
}
