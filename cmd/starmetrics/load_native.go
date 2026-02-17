//go:build !purego && !js

package main

import (
	"fmt"

	"gocv.io/x/gocv"

	sm "starmetrics/pkg/starmetrics"
)

func loadNonFitsImage(path string) (sm.Mat, int, int, error) {
	src := gocv.IMRead(path, gocv.IMReadUnchanged)
	if src.Empty() {
		return sm.Mat{}, 0, 0, fmt.Errorf("could not load image: %s", path)
	}
	defer src.Close()

	w, h := src.Cols(), src.Rows()

	// Convert to float32 [0, 1]
	floatMat := gocv.NewMat()
	src.ConvertTo(&floatMat, gocv.MatTypeCV32F)

	// Scale to [0, 1]
	data, _ := floatMat.DataPtrFloat32()
	n := floatMat.Rows() * floatMat.Cols()
	maxVal := float32(65535.0)
	for i := 0; i < n; i++ {
		data[i] = data[i] / maxVal
	}

	// Wrap in our Mat type via ToFloat32Mat workaround:
	// Extract pixel data and re-create as our Mat
	pixels := make([]uint16, n)
	srcData, _ := src.DataPtrUint16()
	copy(pixels, srcData[:n])
	floatMat.Close()

	return sm.ToFloat32Mat(pixels, 16, w, h), w, h, nil
}
