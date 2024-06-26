package catvnoncat

import (
	"fmt"
	"godl/ml"
	"time"
)

func LLayerModel(X [][]float64, Y [][]float64, layerDims []int64, learningRate float64, numIterations int64) (map[int][][]float64, map[int][]float64) {
	W, b := ml.InitializeParametersDeep(layerDims)

	start := time.Now()
	for i := range numIterations {

		AL, caches := ml.LModelForward(X, W, b)

		fmt.Println("")
		for i := range AL[0] {
			if AL[0][i] > 0.5 {
				fmt.Print(1)
			} else {
				fmt.Print(0)
			}
		}

		fmt.Println("")
		for i := range AL[0] {
			corrects := 0
			if (AL[0][i] > 0.5 && Y[0][i] > 0.5) || (AL[0][i] < 0.5 && Y[0][i] < 0.5) {
				corrects++
				fmt.Print(" ")
			} else {
				fmt.Print("x")
			}
		}

		cost := ml.ComputeCost2d(AL, Y)

		_, dW, db := ml.LModelBackward(AL, Y, caches)

		W, b = ml.UpdateParameters(W, b, dW, db, learningRate)

		if i%10 == 0 {
			fmt.Printf("\nCost after iteration %v: %f -- %s", i, cost, time.Since(start))
			start = time.Now()
		}

	}

	return W, b

}

func Predict(X [][]float64, Y []float64, W map[int][][]float64, b map[int][]float64) []int {
	probs, _ := ml.LModelForward(X, W, b)

	preds := make([]int, len(Y))

	fmt.Println(probs)

	for i := range probs[0] {
		if probs[0][i] > 0.5 {
			preds[i] = 1
		} else {
			preds[i] = 0
		}
	}

	correct := 0
	for i := range preds {
		if preds[i] == int(Y[i]) {
			correct++
		}
	}

	print("Accuracy: ", float64(correct)/float64(len(Y)))

	return preds
}
