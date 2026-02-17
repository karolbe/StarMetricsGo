//go:build purego || js

package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	sm "starmetrics/pkg/starmetrics"
)

func loadNonFitsImage(path string) (sm.Mat, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return sm.Mat{}, 0, 0, fmt.Errorf("opening image: %w", err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return sm.Mat{}, 0, 0, fmt.Errorf("decoding image: %w", err)
	}

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	pixels := make([]uint16, w*h)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			// Convert to grayscale luminance (uint16 range)
			gray := uint16((19595*r + 38470*g + 7471*b + 1<<15) >> 16)
			pixels[y*w+x] = gray
		}
	}

	return sm.ToFloat32Mat(pixels, 16, w, h), w, h, nil
}
