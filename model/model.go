package model

type ForwardPropCache struct {
	APrev [][]float64
	W     [][]float64
	B     []float64
	Z     [][]float64
}
