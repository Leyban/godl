package ml

import (
	"godl/activation"
	"math/rand"
)

type ForwardPropCache struct {
	A [][]float64
	W [][]float64
	B []float64
	Z [][]float64
}

// Matrix Dot Product
func Dot(A, B [][]float64) [][]float64 {

	P := make([][]float64, len(A))

	for i := range P {
		P[i] = make([]float64, len(B[0]))
	}

	for i := range A {
		for j := range B[0] {
			for k := range B {
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

// Linear Forward propagation for single layer
func LinearForward(APrev, W [][]float64, b []float64) (Z [][]float64) {

	Z = Dot(W, APrev)

	for n := range Z {
		for m := range Z[n] {
			Z[n][m] += b[n]
		}
	}

	return Z
}

// Forward Propagation with activation for single layer
func LinearActivationForward(
	APrev, W [][]float64,
	b []float64,
	activ activation.ActivationFunction,
) (A [][]float64, cache ForwardPropCache) {
	Z := LinearForward(APrev, W, b)

	switch activ {
	case activation.ACSigmoid:
		A = activation.Sigmoid(Z)
	case activation.ACReLU:
		A = activation.ReLU(Z)
	}

	cache = ForwardPropCache{
		A: APrev,
		W: W,
		B: b,
		Z: Z,
	}

	return A, cache
}

// Forward Propagation for all Layers
// ReLU -> ReLU -> ... Sigmoid
func LModelForward(
	X [][]float64,
	weights map[int][][]float64,
	biases map[int][]float64,
) (AL [][]float64, caches []ForwardPropCache) {
	A := X
	L := len(weights)

	for l := 1; l < L; l++ {
		APrev := A
		cache := ForwardPropCache{}

		A, cache = LinearActivationForward(APrev, weights[l], biases[l], activation.ACReLU)
		caches = append(caches, cache)
	}

	AL, cache := LinearActivationForward(A, weights[L], biases[L], activation.ACSigmoid)
	caches = append(caches, cache)

	return AL, caches
}
