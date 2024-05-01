package activation

import "math"

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
