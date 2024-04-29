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
		{-1.02387576, 1.12397796},
		{-1.62328545, 0.64667545},
		{-1.74314104, -0.59664964},
	}
	W := [][]float64{
		{0.74505627, 1.97611078, -1.24412333},
	}
	b := []float64{1}

	Z := ml.LinearForward(A, W, b)
	fmt.Println(Z)
}
