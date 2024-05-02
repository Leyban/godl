package ml

// LinearBackward implements the linear portion of backward propagation for a single layer (layer l).
func LinearBackward(dZ [][]float64, cache ForwardPropCache) (dAprev, dW [][]float64, db []float64) {
	Aprev, W := cache.A, cache.W
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
