package catvnoncat

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
)

func visualizeImage(data [12288]uint8) {

	// Create a new image with the same dimensions
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))

	// Fill the image with the RGB data
	for i := 0; i < 64; i++ {
		for j := 0; j < 64; j++ {
			index := i*64*3 + j*3

			img.Set(j, i, color.RGBA{
				R: data[index],
				G: data[index+1],
				B: data[index+2],
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
