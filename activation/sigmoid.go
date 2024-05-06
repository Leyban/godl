package activation

import (
	"godl/model"
	"math"
)

func Sigmoid(Z [][]float64) (A [][]float64) {
	A = make([][]float64, len(Z))
	for m := range A {
		A[m] = make([]float64, len(Z[m]))
	}

	for m := range A {
		for n := range A[m] {
			A[m][n] = 1 / (1 + math.Exp(-Z[m][n]))
		}
	}

	return A
}

// calculate the g'(z) for Sigmoid
// dA[n][m]
// dZ[n][m]
// g'(z) = g(z) * (1 - g(z))
func SigmoidBackward(dA [][]float64, cache model.ForwardPropCache) (dZ [][]float64) {
	dZ = make([][]float64, len(dA))

	for n := range dA {
		dZ[n] = make([]float64, len(dA[n]))
		for m := range dA[n] {
			s := 1 / (1 + math.Exp(-cache.Z[n][m]))
			dZ[n][m] = dA[n][m] * s * (1 - s)
		}
	}

	return dZ
}
