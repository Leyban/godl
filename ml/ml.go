package ml

import (
	"fmt"
	"math/rand"
)

// Matrix Dot Product
func Dot(A, B [][]float64) [][]float64 {

	P := make([][]float64, len(A))

	for i := range P {
		P[i] = make([]float64, len(B))
	}

	fmt.Println(P)
	fmt.Println(len(A), len(A[0]))
	fmt.Println(len(B), len(B[0]))

	for i := 0; i < len(A); i++ {
		for j := 0; j < len(B[0]); j++ {
			for k := 0; k < len(B); k++ {
				fmt.Printf("P%v%v ", i, j)
				fmt.Printf("A%v%v ", i, k)
				fmt.Printf("W%v%v ", k, j)
				fmt.Print("\n")
				P[i][j] += A[i][k] * B[k][j]
			}
		}
	}

	return P
}

// params sizes -- input, hidden, output
// output weights, biases
func InitializeParameters(n_x, n_h, n_y int64) (map[string][][]float64, map[string][]float64) {

	W1 := make([][]float64, n_h)
	for i := range W1 {
		W1[i] = make([]float64, n_x)

		for j := range W1[i] {
			W1[i][j] = rand.Float64() * 0.01
		}
	}

	b1 := make([]float64, n_h)

	W2 := make([][]float64, n_y)
	for i := range W2 {
		W2[i] = make([]float64, n_h)

		for j := range W2[i] {
			W2[i][j] = rand.Float64() * 0.01
		}
	}

	b2 := make([]float64, n_y)

	weights := map[string][][]float64{
		"W1": W1,
		"W2": W2,
	}

	biases := map[string][]float64{
		"b1": b1,
		"b2": b2,
	}

	return weights, biases
}

// Initialize Weights and Biases based on input layerDims
// layerDims is the array of number of nodes per layer
func InitializeParametersDeep(layerDims []int64) (map[int][][]float64, map[int][]float64) {

	weights := make(map[int][][]float64)
	biases := make(map[int][]float64)

	for l := 1; l < len(layerDims); l++ {
		weights[l] = make([][]float64, layerDims[l])

		for i := range weights[l] {
			weights[l][i] = make([]float64, layerDims[l-1])

			for j := range weights[l][i] {
				weights[l][i][j] = rand.Float64() * 0.01
			}

		}

		biases[l] = make([]float64, layerDims[l])
	}

	return weights, biases
}

// Forward propagation
func LinearForward(A, W [][]float64, b []float64) [][]float64 {

	Z := Dot(A, W)

	for i := range Z {
		for j := range Z[i] {
			Z[i][j] += b[j]
		}
	}

	return Z
}
