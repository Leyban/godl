package ml

import (
	"godl/activation"
	"godl/model"
)

/*
LinearBackward implements the linear portion of backward propagation for a single layer (layer l).

	    dZ:     (n, m)
		APrev:  (n-1, m)
		W:      (n, n-1)
		b:      (n,)

		dAprev: (n-1, m)
		dW:     (n, n-1)
		db:     (n,)
*/
func LinearBackward(dZ [][]float64, cache model.ForwardPropCache) (dAprev, dW [][]float64, db []float64) {
	Aprev, W := cache.APrev, cache.W
	m := len(Aprev[0])

	// Get dW
	dW = Dot(dZ, Transpose(Aprev))
	for i := range dW {
		for j := range dW[i] {
			dW[i][j] /= float64(m)
		}
	}

	// Get db
	db = make([]float64, len(dZ))
	for i := range db {
		for j := range dZ[i] {
			db[i] += dZ[i][j]
		}
	}
	for i := range db {
		db[i] /= float64(m)
	}

	// Get dAprev
	dAprev = Dot(Transpose(W), dZ)

	return dAprev, dW, db
}

func LinearActivationBackward(dA [][]float64, cache model.ForwardPropCache, activ activation.ActivationFunction) (dAPrev, dW [][]float64, db []float64) {
	var dZ [][]float64

	if activ == activation.ACReLU {
		dZ = activation.ReLUBAckward(dA, cache)
	} else if activ == activation.ACSigmoid {
		dZ = activation.SigmoidBackward(dA, cache)
	}

	dAPrev, dW, db = LinearBackward(dZ, cache)

	return dAPrev, dW, db
}

func LModelBackward(AL [][]float64, Y [][]float64, caches []model.ForwardPropCache) (dA map[int][][]float64, dW map[int][][]float64, db map[int][]float64) {
	L := len(caches)

	dAL := make([][]float64, len(AL))
	for n := range AL {
		dAL[n] = make([]float64, len(AL[n]))

		for m := range AL[n] {
			dAL[n][m] = -((Y[n][m] / AL[n][m]) - ((1 - Y[n][m]) / (1 - AL[n][m])))
		}
	}

	dA = make(map[int][][]float64)
	dW = make(map[int][][]float64)
	db = make(map[int][]float64)

	dA[L] = dAL

	cache := caches[L-1]
	dA[L-1], dW[L], db[L] = LinearActivationBackward(dAL, cache, activation.ACSigmoid)

	for l := L - 1; l > 0; l-- {
		cache := caches[l-1]
		dA[l-1], dW[l], db[l] = LinearActivationBackward(dA[l], cache, activation.ACReLU)
	}

	return dA, dW, db
}
