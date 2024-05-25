package ml

import (
	"math/rand"
)

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
// weights[n][n-1] biases[n]
func InitializeParametersDeep(layerDims []int64) (weights map[int][][]float64, biases map[int][]float64) {

	weights = make(map[int][][]float64)
	biases = make(map[int][]float64)

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

func UpdateParameters(
	weights map[int][][]float64,
	biases map[int][]float64,

	dW map[int][][]float64,
	db map[int][]float64,

	learning_rate float64,
) (
	map[int][][]float64,
	map[int][]float64,
) {
	for l := range weights {
		for n := range weights[l] {
			for m := range weights[l][n] {
				weights[l][n][m] = weights[l][n][m] - (learning_rate * dW[l][n][m])
			}
			biases[l][n] = biases[l][n] - (learning_rate * db[l][n])
		}
	}

	return weights, biases
}
