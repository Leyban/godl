package ml_test

import (
	"godl/activation"
	"godl/ml"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDot(t *testing.T) {
	A := [][]float64{
		{1, 2},
		{3, 4},
	}
	B := [][]float64{
		{5, 6},
		{7, 8},
	}
	expected := [][]float64{
		{19, 22},
		{43, 50},
	}
	actual := ml.Dot(A, B)
	assert.Equal(t, expected, actual)
}

func TestInitializeParameters(t *testing.T) {
	n_x := int64(2)
	n_h := int64(3)
	n_y := int64(1)
	weights, biases := ml.InitializeParameters(n_x, n_h, n_y)
	assert.Equal(t, 3, len(weights))
	assert.Equal(t, 2, len(biases))
	assert.Equal(t, 3, len(weights["W1"]))
	assert.Equal(t, 2, len(weights["W1"][0]))
	assert.Equal(t, 1, len(weights["W2"]))
	assert.Equal(t, 3, len(weights["W2"][0]))
	assert.Equal(t, 3, len(biases["b1"]))
	assert.Equal(t, 1, len(biases["b2"]))
}

func TestInitializeParametersDeep(t *testing.T) {
	layerDims := []int64{2, 3, 1}
	weights, biases := ml.InitializeParametersDeep(layerDims)
	assert.Equal(t, 2, len(weights))
	assert.Equal(t, 2, len(biases))
	assert.Equal(t, 3, len(weights[1]))
	assert.Equal(t, 2, len(weights[1][0]))
	assert.Equal(t, 1, len(weights[2]))
	assert.Equal(t, 3, len(weights[2][0]))
	assert.Equal(t, 3, len(biases[1]))
	assert.Equal(t, 1, len(biases[2]))
}

func TestLinearForward(t *testing.T) {
	APrev := [][]float64{
		{1, 2},
		{3, 4},
	}
	W := [][]float64{
		{5, 6},
		{7, 8},
	}
	b := []float64{9, 10}
	expected := [][]float64{
		{29, 34},
		{53, 62},
	}
	actual := ml.LinearForward(APrev, W, b)
	assert.Equal(t, expected, actual)
}

func TestLinearActivationForward(t *testing.T) {
	APrev := [][]float64{
		{1, 2},
		{3, 4},
	}
	W := [][]float64{
		{5, 6},
		{7, 8},
	}
	b := []float64{9, 10}
	expected := [][]float64{
		{0.9933071490758151, 0.9975273769815209},
		{0.9990889485042811, 0.9999042791896507},
	}
	actual, _ := ml.LinearActivationForward(APrev, W, b, activation.ACSigmoid)
	assert.Equal(t, expected, actual)
}

func TestLModelForward(t *testing.T) {
	X := [][]float64{
		{1, 2},
		{3, 4},
	}
	weights := map[int][][]float64{
		1: [][]float64{
			{5, 6},
			{7, 8},
		},
		2: [][]float64{
			{9, 10},
			{11, 12},
		},
	}
	biases := map[int][]float64{
		1: []float64{9, 10},
		2: []float64{11, 12},
	}
	expected := [][]float64{
		{0.9933071490758151, 0.9975273769815209},
		{0.9990889485042811, 0.9999042791896507},
	}
	actual, _ := ml.LModelForward(X, weights, biases)
	assert.Equal(t, expected, actual)
}
