package main

import (
	"fmt"
	"godl/activation"
	"godl/ml"
	"godl/model"
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

	A, _ := ml.LinearActivationForward(APrev, W, b, activation.ACSigmoid)
	fmt.Println(A)

	A, _ = ml.LinearActivationForward(APrev, W, b, activation.ACReLU)
	fmt.Println(A)

}

func test4() {
	X := [][]float64{
		{-0.31178367, 0.72900392, 0.21782079, -0.8990918},
		{-2.48678065, 0.91325152, 1.12706373, -1.51409323},
		{1.63929108, -0.4298936, 2.63128056, 0.60182225},
		{-0.33588161, 1.23773784, 0.11112817, 0.12915125},
		{0.07612761, -0.15512816, 0.63422534, 0.810655},
	}
	weights := map[int][][]float64{
		1: {
			{0.35480861, 1.81259031, -1.3564758, -0.46363197, 0.82465384},
			{-1.17643148, 1.56448966, 0.71270509, -0.1810066, 0.53419953},
			{-0.58661296, -1.48185327, 0.85724762, 0.94309899, 0.11444143},
			{-0.02195668, -2.12714455, -0.83440747, -0.46550831, 0.23371059},
		},
		2: {
			{-0.12673638, -1.36861282, 1.21848065, -0.85750144},
			{-0.56147088, -1.0335199, 0.35877096, 1.07368134},
			{-0.37550472, 0.39636757, -0.47144628, 2.33660781}},
		3: {{0.9398248, 0.42628539, -0.75815703}},
	}

	biases := map[int][]float64{
		1: {
			1.38503523,
			-0.51962709,
			-0.78015214,
			0.95560959,
		},
		2: {
			1.50278553,
			-0.59545972,
			0.52834106,
		},
		3: {-0.16236698},
	}

	AL, _ := ml.LModelForward(X, weights, biases)

	fmt.Println(AL)
}

func test5() {
	Y := [][]float64{{1, 1, 0}}
	AL := [][]float64{{.8, .9, 0.4}}

	cost := ml.ComputeCost(AL, Y)
	fmt.Println(cost)

}

func test6() {
	dZ := [][]float64{
		{1.62434536, -0.61175641, -0.52817175, -1.07296862},
		{0.86540763, -2.3015387, 1.74481176, -0.7612069},
		{0.3190391, -0.24937038, 1.46210794, -2.06014071},
	}
	cache := model.ForwardPropCache{
		APrev: [][]float64{
			{-0.3224172, -0.38405435, 1.13376944, -1.09989127},
			{-0.17242821, -0.87785842, 0.04221375, 0.58281521},
			{-1.10061918, 1.14472371, 0.90159072, 0.50249434},
			{0.90085595, -0.68372786, -0.12289023, -0.93576943},
			{-0.26788808, 0.53035547, -0.69166075, -0.39675353},
		},
		W: [][]float64{
			{-0.6871727, -0.84520564, -0.67124613, -0.0126646, -1.11731035},
			{0.2344157, 1.65980218, 0.74204416, -0.19183555, -0.88762896},
			{-0.74715829, 1.6924546, 0.05080775, -0.63699565, 0.19091548},
		},
	}

	dA_prev, dW, db := ml.LinearBackward(dZ, cache)
	fmt.Println(dA_prev)
	fmt.Println("---------------------------")
	fmt.Println(dW)
	fmt.Println("---------------------------")
	fmt.Println(db)

}

func test7() {
	dAL := [][]float64{{-0.41675785, -0.05626683}}
	cache := model.ForwardPropCache{
		Z: [][]float64{{0.04153939, -1.11792545}},
		APrev: [][]float64{
			{-2.1361961, 1.64027081},
			{-1.79343559, -0.84174737},
			{0.50288142, -1.24528809},
		},
		W: [][]float64{{-1.05795222, -0.90900761, 0.55145404}},
		B: []float64{2.29220801},
	}

	fmt.Println("sigmoid")
	dAPrev, dW, db := ml.LinearActivationBackward(dAL, cache, activation.ACSigmoid)
	fmt.Println("dAPrev", dAPrev)
	fmt.Println("dW", dW)
	fmt.Println("db", db)

	fmt.Println("relu")
	dAPrev, dW, db = ml.LinearActivationBackward(dAL, cache, activation.ACReLU)
	fmt.Println("dAPrev", dAPrev)
	fmt.Println("dW", dW)
	fmt.Println("db", db)

}
