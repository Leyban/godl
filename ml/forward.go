package ml

import (
	"godl/activation"
	"godl/model"
	"math"
)

// Linear Forward propagation for single layer
// APrev[m][n]
// W[n][n-1]
// b[n]
// Z[m][n]
func LinearForward(APrev, W [][]float64, b []float64) (Z [][]float64) {

	Z = Dot(W, APrev)

	for i := range Z {
		for j := range Z[i] {
			Z[i][j] += b[i]
		}
	}

	return Z
}

// Forward Propagation with activation for single layer
func LinearActivationForward(
	APrev, W [][]float64,
	b []float64,
	activ activation.ActivationFunction,
) (A [][]float64, cache model.ForwardPropCache) {
	Z := LinearForward(APrev, W, b)

	switch activ {
	case activation.ACSigmoid:
		A = activation.Sigmoid(Z)
	case activation.ACReLU:
		A = activation.ReLU(Z)
	}

	cache = model.ForwardPropCache{
		APrev: APrev,
		W:     W,
		B:     b,
		Z:     Z,
	}

	return A, cache
}

// Forward Propagation for all Layers
// ReLU -> ReLU -> ... Sigmoid
func LModelForward(
	X [][]float64,
	weights map[int][][]float64,
	biases map[int][]float64,
) (AL [][]float64, caches []model.ForwardPropCache) {
	A := X
	L := len(weights)

	for l := 1; l < L; l++ {
		APrev := A
		cache := model.ForwardPropCache{}

		A, cache = LinearActivationForward(APrev, weights[l], biases[l], activation.ACReLU)
		caches = append(caches, cache)
	}

	AL, cache := LinearActivationForward(A, weights[L], biases[L], activation.ACSigmoid)
	caches = append(caches, cache)

	return AL, caches
}

// Compute loss from AL and Y.
// Only works on single array AL and single array Y
func ComputeCost2d(AL, Y [][]float64) float64 {
	m := len(Y[0])

	var cost float64
	for i := 0; i < m; i++ {
		cost += math.Abs(-Y[0][i]*math.Log(AL[0][i]) - (1-Y[0][i])*math.Log(1-AL[0][i]))
	}
	cost /= float64(m)

	return cost
}

func ComputeCost1d(AL, Y []float64) float64 {
	m := len(Y)

	var cost float64
	for i := 0; i < m; i++ {
		cost += -Y[i]*math.Log(AL[i]) - (1-Y[i])*math.Log(1-AL[i])
	}
	cost /= float64(m)

	return cost
}
