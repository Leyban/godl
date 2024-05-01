package main

import (
	"fmt"
	"godl/activation"
	"godl/ml"
)

func test1() {
	weights, biases := ml.InitializeParametersDeep([]int64{5, 4, 3})

	fmt.Println(weights)
	fmt.Println(biases)
}

func test2() {

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

func test3() {

	APrev := [][]float64{
		{-0.41675785, -0.05626683},
		{-2.1361961, 1.64027081},
		{-1.79343559, -0.84174737},
	}
	W := [][]float64{
		{0.50288142, -1.24528809, -1.05795222},
	}
	b := []float64{-0.90900761}

	A := ml.LinearActivationForward(APrev, W, b, activation.ACSigmoid)
	fmt.Println(A)

	A = ml.LinearActivationForward(APrev, W, b, activation.ACReLU)
	fmt.Println(A)

}
