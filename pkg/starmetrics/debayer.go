package starmetrics

// DebayerRGGB performs bilinear interpolation on a raw RGGB Bayer-pattern image
// and returns a luminance channel: (R + G + B) / 3 per pixel.
//
// RGGB layout (row-major, 0-indexed):
//
//	(even row, even col) = R
//	(even row, odd  col) = G  (Gr)
//	(odd  row, even col) = G  (Gb)
//	(odd  row, odd  col) = B
//
// Edge pixels use clamped (replicated) neighbor lookups.
func DebayerRGGB(data []float64, width, height int) []float64 {
	out := make([]float64, width*height)

	// clamp helpers
	clampX := func(x int) int {
		if x < 0 {
			return 0
		}
		if x >= width {
			return width - 1
		}
		return x
	}
	clampY := func(y int) int {
		if y < 0 {
			return 0
		}
		if y >= height {
			return height - 1
		}
		return y
	}
	px := func(x, y int) float64 {
		return data[clampY(y)*width+clampX(x)]
	}

	for y := 0; y < height; y++ {
		evenRow := y%2 == 0
		for x := 0; x < width; x++ {
			evenCol := x%2 == 0
			var r, g, b float64

			switch {
			case evenRow && evenCol:
				// Red pixel — have R, need G and B
				r = px(x, y)
				g = (px(x-1, y) + px(x+1, y) + px(x, y-1) + px(x, y+1)) / 4
				b = (px(x-1, y-1) + px(x+1, y-1) + px(x-1, y+1) + px(x+1, y+1)) / 4

			case evenRow && !evenCol:
				// Green on red row (Gr) — need R and B
				r = (px(x-1, y) + px(x+1, y)) / 2
				g = px(x, y)
				b = (px(x, y-1) + px(x, y+1)) / 2

			case !evenRow && evenCol:
				// Green on blue row (Gb) — need R and B
				r = (px(x, y-1) + px(x, y+1)) / 2
				g = px(x, y)
				b = (px(x-1, y) + px(x+1, y)) / 2

			default:
				// Blue pixel — have B, need R and G
				r = (px(x-1, y-1) + px(x+1, y-1) + px(x-1, y+1) + px(x+1, y+1)) / 4
				g = (px(x-1, y) + px(x+1, y) + px(x, y-1) + px(x, y+1)) / 4
				b = px(x, y)
			}

			out[y*width+x] = (r + g + b) / 3
		}
	}

	return out
}

// DebayerToMat converts raw uint16 Bayer-pattern pixels to a debayered float32 Mat.
func DebayerToMat(pixels []uint16, bitDepth, width, height int) Mat {
	maxVal := float64(uint64(1)<<uint(bitDepth) - 1)
	data := make([]float64, len(pixels))
	for i, p := range pixels {
		data[i] = float64(p) / maxVal
	}
	lum := DebayerRGGB(data, width, height)
	mat := NewMatWithSize(height, width)
	dest := mat.DataFloat32()
	for i, v := range lum {
		dest[i] = float32(v)
	}
	return mat
}
