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

	// dZ := make([][]float64, len(dA))
	// for i := range dA {
	// 	dZ[i] = make([]float64, len(dA[i]))
	// }

	if activ == activation.ACReLU {
		dZ = activation.ReLUBAckward(dA, cache)
	} else if activ == activation.ACSigmoid {
		dZ = activation.SigmoidBackward(dA, cache)
	}

	dAPrev, dW, db = LinearBackward(dZ, cache)

	return dAPrev, dW, db
}
