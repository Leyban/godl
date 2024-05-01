package activation

import "math"

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
