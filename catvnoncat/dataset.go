package catvnoncat

import (
	"log"

	"gonum.org/v1/hdf5"
)

func LoadTrainData() ([][]float64, []float64) {
	// Open the HDF5 file
	file, err := hdf5.OpenFile("dataset/train_catvnoncat.h5", hdf5.F_ACC_RDONLY)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Open the trainSetX
	trainSetX, err := file.OpenDataset("train_set_x")
	if err != nil {
		log.Fatal(err)
	}
	defer trainSetX.Close()

	numItems := 209

	// Allocate a contiguous block of memory for the dataX
	dataX := make([]uint8, numItems*64*64*3) // Assuming the data is uint8

	err = trainSetX.Read(&dataX)
	if err != nil {
		log.Fatal(err)
	}

	// // Forge the inputs into image buildable map
	// reshapedDataX := make([][12288]uint8, numItems)
	// for i := 0; i < numItems; i++ {
	// 	for j := 0; j < 64*64*3; j++ {
	// 		reshapedDataX[i][j] = dataX[i*64*64*3+j]
	// 	}
	// }

	// Forge the inputs A0
	X := make([][]float64, 64*64*3)
	for i := 0; i < 64*64*3; i++ {
		X[i] = make([]float64, numItems)
		for j := 0; j < numItems; j++ {
			X[i][j] = float64(dataX[i+j*numItems])
		}
	}

	// Open the trainSetY
	trainSetY, err := file.OpenDataset("train_set_y")
	if err != nil {
		log.Fatal(err)
	}
	defer trainSetY.Close()

	// Allocate a contiguous block of memory for the dataY
	dataY := make([]int, numItems)

	err = trainSetY.Read(&dataY)
	if err != nil {
		log.Fatal(err)
	}

	Y := make([]float64, numItems)
	for i := 0; i < numItems; i++ {
		Y[i] = float64(dataY[i])
	}

	return X, Y
}

func LoadTestData() ([][]float64, []float64) {
	// Open the HDF5 file
	file, err := hdf5.OpenFile("dataset/test_catvnoncat.h5", hdf5.F_ACC_RDONLY)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Open the testSetX
	testSetX, err := file.OpenDataset("test_set_x")
	if err != nil {
		log.Fatal(err)
	}
	defer testSetX.Close()

	numItems := 50

	// Allocate a contiguous block of memory for the dataX
	dataX := make([]uint8, numItems*64*64*3) // Assuming the data is uint8

	err = testSetX.Read(&dataX)
	if err != nil {
		log.Fatal(err)
	}

	// // Forge the inputs into image buildable map
	// reshapedDataX := make([][12288]uint8, numItems)
	// for i := 0; i < numItems; i++ {
	// 	for j := 0; j < 64*64*3; j++ {
	// 		reshapedDataX[i][j] = dataX[i*64*64*3+j]
	// 	}
	// }

	// Forge the inputs A0
	X := make([][]float64, 64*64*3)
	for i := 0; i < 64*64*3; i++ {
		X[i] = make([]float64, numItems)
		for j := 0; j < numItems; j++ {
			X[i][j] = float64(dataX[i+j*numItems])
		}
	}

	// Open the testSetY
	testSetY, err := file.OpenDataset("test_set_y")
	if err != nil {
		log.Fatal(err)
	}
	defer testSetY.Close()

	// Allocate a contiguous block of memory for the dataY
	dataY := make([]int, numItems)

	err = testSetY.Read(&dataY)
	if err != nil {
		log.Fatal(err)
	}

	Y := make([]float64, numItems)
	for i := 0; i < numItems; i++ {
		Y[i] = float64(dataY[i])
	}

	return X, Y
}
