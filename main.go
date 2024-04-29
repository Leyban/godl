package main

import (
	"fmt"
	"godl/ml"
)

func main() {
	// weights, biases := ml.InitializeParametersDeep([]int64{5, 4, 3})

	// fmt.Println(weights)
	// fmt.Println(biases)

	A := [][]float64{
		{1.62434536, -0.61175641},
		{-0.52817175, -1.07296862},
		{0.86540763, -2.3015387},
	}
	W := [][]float64{
		{1.74481176, -0.7612069, 0.3190391},
	}
	b := []float64{-0.24937038}

	Z := ml.LinearForward(A, W, b)
	fmt.Println(Z)
}
