package activation

import "math"

// Rectified Linear Unit Function
// Args:
// Z -- Outputs of the Linear Layer
func Relu(Z []float64) (A []float64) {
	A = make([]float64, len(Z))

	for i := range A {
		A[i] = math.Max(0, Z[i])
	}

	return A
}
