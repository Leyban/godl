package main

import (
	"fmt"
	"godl/catvnoncat"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"reflect"

	"gonum.org/v1/hdf5"
)

func test11() {
	// Open the HDF5 file
	file, err := hdf5.OpenFile("dataset/test_catvnoncat.h5", hdf5.F_ACC_RDONLY)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	dataset, err := file.OpenDataset("test_set_y")
	if err != nil {
		log.Fatal(err)
	}
	defer dataset.Close()

	fmt.Println(dataset)

	data := make([]int64, 50)

	err = dataset.Read(&data)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(data)
}

func test12() {
	// Open the HDF5 file
	file, err := hdf5.OpenFile("dataset/test_catvnoncat.h5", hdf5.F_ACC_RDONLY)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Open the dataset
	dataset, err := file.OpenDataset("test_set_x")
	if err != nil {
		log.Fatal(err)
	}
	defer dataset.Close()

	type xtype struct {
	}

	// Allocate a contiguous block of memory for the data
	data := make([]float64, 50*64*64*3)

	// Read the dataset
	err = dataset.Read(&data)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(data)
}

func test13() {
	// Open the HDF5 file
	file, err := hdf5.OpenFile("dataset/train_catvnoncat.h5", hdf5.F_ACC_RDONLY)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Open the dataset
	dataset, err := file.OpenDataset("train_set_x")
	if err != nil {
		log.Fatal(err)
	}
	defer dataset.Close()

	numItems := 209

	// Allocate a contiguous block of memory for the data
	data := make([]uint8, numItems*64*64*3) // Assuming the data is uint8

	// Read the dataset
	err = dataset.Read(&data)
	if err != nil {
		log.Fatal(err)
	}

	// Reshape the data into a 4D slice of RGB
	reshapedData := make([][64][64][3]uint8, numItems)
	for i := 0; i < numItems; i++ {
		for j := 0; j < 64; j++ {
			for k := 0; k < 64; k++ {
				index := (i*64*64*3 + j*64*3 + k*3)
				reshapedData[i][j][k][0] = data[index]
				reshapedData[i][j][k][1] = data[index+1]
				reshapedData[i][j][k][2] = data[index+2]
			}
		}
	}

	// Visualize the first image
	visualizeImage(reshapedData[16])
}

func visualizeImage(data [64][64][3]uint8) {
	// Create a new image with the same dimensions
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))

	// Fill the image with the RGB data
	for i := 0; i < 64; i++ {
		for j := 0; j < 64; j++ {
			pixel := data[i][j]
			img.Set(j, i, color.RGBA{
				R: pixel[0],
				G: pixel[1],
				B: pixel[2],
				A: 255,
			})
		}
	}

	// Save the image to a file
	file, err := os.Create("output.png")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	png.Encode(file, img)
	fmt.Println("Image saved to output.png")
}

func test14() {
	// Open the HDF5 file
	file, err := hdf5.OpenFile("dataset/train_catvnoncat.h5", hdf5.F_ACC_RDONLY)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Open the dataset
	dataset, err := file.OpenDataset("train_set_y")
	if err != nil {
		log.Fatal(err)
	}
	defer dataset.Close()

	numItems := 209
	// Allocate a contiguous block of memory for the data
	data := make([]int, numItems) // Assuming the data is uint8

	// Read the dataset
	err = dataset.Read(&data)
	if err != nil {
		log.Fatal(err)
	}

	for i := range data {
		if data[i] == 1 {
			fmt.Println(i, data[i])
		}
	}
}

func test15() {
	X, Y := catvnoncat.LoadTrainData()
	fmt.Println(reflect.TypeOf(X), len(X), len(X[0]))
	fmt.Println(reflect.TypeOf(Y), len(Y))
}
