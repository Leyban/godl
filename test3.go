package main

import (
	"fmt"
	"godl/catvnoncat"
	"time"
)

func test16() {
	X_train, Y_train_raw := catvnoncat.LoadTrainData()

	Y_train := make([][]float64, 1)
	Y_train[0] = Y_train_raw

	layerDims := []int64{12288, 20, 7, 5, 1}
	learningRate := 0.0075

	var numIterations int64 = 2500

	start := time.Now()

	W, b := catvnoncat.LLayerModel(X_train, Y_train, layerDims, learningRate, numIterations)

	fmt.Println("Total Time: ", time.Since(start))

	fmt.Println(W, b)
}

func test17() {
	X_train, Y_train_raw := catvnoncat.LoadTrainData()

	Y_train := make([][]float64, 1)
	Y_train[0] = Y_train_raw

	layerDims := []int64{12288, 20, 7, 5, 1}
	learningRate := 0.0075

	var numIterations int64 = 2500

	start := time.Now()

	W, b := catvnoncat.LLayerModel(X_train, Y_train, layerDims, learningRate, numIterations)

	fmt.Println("\nTotal Time: ", time.Since(start))

	X_test, Y_test := catvnoncat.LoadTestData()

	preds := catvnoncat.Predict(X_test, Y_test, W, b)

	fmt.Println(Y_test)
	fmt.Println(preds)

	catness := map[int]string{
		1: "cat",
		0: "not cat",
	}

	for i := range preds {
		if preds[i] != int(Y_test[i]) {
			fmt.Println(i, "looks like", catness[preds[i]])
		}
	}

}
