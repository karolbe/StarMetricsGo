package starmetrics

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// RenderFieldOverlay generates a JPG image showing the tilt analysis visually and writes it to a file.
func RenderFieldOverlay(field *FieldAnalysis, width, height int, outputPath string) error {
	img, err := renderFieldImage(field, width, height)
	if err != nil {
		return err
	}

	f, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create overlay file: %w", err)
	}
	defer f.Close()

	return jpeg.Encode(f, img, &jpeg.Options{Quality: 90})
}

// RenderFieldOverlayBytes generates a JPG image showing the tilt analysis and returns it as JPEG bytes.
func RenderFieldOverlayBytes(field *FieldAnalysis, width, height int) ([]byte, error) {
	img, err := renderFieldImage(field, width, height)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// renderFieldImage creates the overlay image in memory.
func renderFieldImage(field *FieldAnalysis, width, height int) (*image.RGBA, error) {
	if field == nil {
		return nil, fmt.Errorf("no field analysis data")
	}

	// Render at reduced resolution (800px wide, proportional height)
	const targetWidth = 800
	scale := float64(targetWidth) / float64(width)
	imgW := targetWidth
	imgH := int(float64(height) * scale)
	if imgH < 100 {
		imgH = 100
	}

	// Reserve space for summary text at bottom
	summaryH := 60
	totalH := imgH + summaryH

	img := image.NewRGBA(image.Rect(0, 0, imgW, totalH))

	// Black background
	for y := 0; y < totalH; y++ {
		for x := 0; x < imgW; x++ {
			img.Set(x, y, color.RGBA{0, 0, 0, 255})
		}
	}

	// Get zone fraction boundaries
	frac := 0.25 // matches default
	xLo := int(float64(imgW) * frac)
	xHi := int(float64(imgW) * (1.0 - frac))
	yLo := int(float64(imgH) * frac)
	yHi := int(float64(imgH) * (1.0 - frac))

	// Zone bounds: [row][col] -> (x0, y0, x1, y1)
	xBounds := [3][2]int{{0, xLo}, {xLo, xHi}, {xHi, imgW}}
	yBounds := [3][2]int{{0, yLo}, {yLo, yHi}, {yHi, imgH}}

	// Zone grid mapping: [row][col] -> ZonePosition
	zoneGrid := [3][3]ZonePosition{
		{ZoneTopLeft, ZoneTop, ZoneTopRight},
		{ZoneLeft, ZoneCenter, ZoneRight},
		{ZoneBottomLeft, ZoneBottom, ZoneBottomRight},
	}

	centerHFR := field.Zones[ZoneCenter].MedianHFR
	if centerHFR <= 0 {
		centerHFR = 1
	}

	// Fill zones with color based on HFR ratio to center
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			zone := field.Zones[zoneGrid[row][col]]
			c := hfrColor(zone.MedianHFR, centerHFR)
			x0, x1 := xBounds[col][0], xBounds[col][1]
			y0, y1 := yBounds[row][0], yBounds[row][1]
			for y := y0; y < y1; y++ {
				for x := x0; x < x1; x++ {
					img.Set(x, y, c)
				}
			}
		}
	}

	// Draw grid lines (white, 1px)
	gridColor := color.RGBA{255, 255, 255, 180}
	for x := 0; x < imgW; x++ {
		img.Set(x, yLo, gridColor)
		img.Set(x, yHi, gridColor)
	}
	for y := 0; y < imgH; y++ {
		img.Set(xLo, y, gridColor)
		img.Set(xHi, y, gridColor)
	}

	// Draw per-zone text labels and HFR circles
	face := basicfont.Face7x13
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			zone := field.Zones[zoneGrid[row][col]]
			x0, x1 := xBounds[col][0], xBounds[col][1]
			y0, y1 := yBounds[row][0], yBounds[row][1]
			cx := (x0 + x1) / 2
			cy := (y0 + y1) / 2

			// Draw HFR circle at zone center
			if zone.MedianHFR > 0 {
				radius := int(zone.MedianHFR * scale * 3)
				if radius < 3 {
					radius = 3
				}
				if radius > (x1-x0)/3 {
					radius = (x1 - x0) / 3
				}
				drawCircle(img, cx, cy, radius, color.RGBA{255, 255, 255, 200})
			}

			// Draw text: zone label, HFR value, star count
			textColor := color.RGBA{255, 255, 255, 255}
			line1 := zone.Label
			line2 := fmt.Sprintf("HFR: %.2f", zone.MedianHFR)
			line3 := fmt.Sprintf("n=%d", zone.StarCount)

			drawCenteredText(img, face, line1, cx, cy-14, textColor)
			drawCenteredText(img, face, line2, cx, cy+2, textColor)
			drawCenteredText(img, face, line3, cx, cy+16, textColor)
		}
	}

	// Draw arrow from best corner to worst corner (tilt direction)
	if field.WorstCorner != "" && field.BestCorner != "" {
		bestX, bestY := cornerCenter(field.BestCorner, xBounds, yBounds)
		worstX, worstY := cornerCenter(field.WorstCorner, xBounds, yBounds)
		arrowColor := color.RGBA{255, 80, 80, 255}
		drawLine(img, bestX, bestY, worstX, worstY, arrowColor)
		drawArrowHead(img, bestX, bestY, worstX, worstY, arrowColor)
	}

	// Summary text at bottom
	summaryColor := color.RGBA{220, 220, 220, 255}
	summaryY := imgH + 15
	tiltStr := fmt.Sprintf("Tilt: %.1f%%  (worst: %s, best: %s)", field.TiltPct, field.WorstCorner, field.BestCorner)
	offAxisStr := fmt.Sprintf("Off-axis: %.1f%%", field.OffAxisPct)
	reliableStr := ""
	if !field.Reliable {
		reliableStr = "  [LOW STAR COUNT - UNRELIABLE]"
	}

	drawText(img, face, tiltStr, 10, summaryY, summaryColor)
	drawText(img, face, offAxisStr+reliableStr, 10, summaryY+18, summaryColor)

	return img, nil
}

// hfrColor returns a color based on the HFR ratio to center.
func hfrColor(zoneHFR, centerHFR float64) color.RGBA {
	if zoneHFR <= 0 || centerHFR <= 0 {
		return color.RGBA{40, 40, 40, 255}
	}
	ratio := zoneHFR / centerHFR

	var r, g, b uint8
	switch {
	case ratio <= 1.1:
		// Green
		t := ratio / 1.1
		g = uint8(60 + t*40)
		r = uint8(t * 30)
		b = 20
	case ratio <= 1.3:
		// Green -> Yellow
		t := (ratio - 1.1) / 0.2
		r = uint8(30 + t*170)
		g = uint8(100 - t*20)
		b = 20
	default:
		// Yellow -> Red
		t := math.Min((ratio-1.3)/0.3, 1.0)
		r = uint8(200 + t*55)
		g = uint8(80 - t*60)
		b = uint8(20 - t*10)
	}
	return color.RGBA{r, g, b, 255}
}

// cornerCenter returns the center pixel coords for a named corner.
func cornerCenter(label string, xBounds [3][2]int, yBounds [3][2]int) (int, int) {
	var col, row int
	switch label {
	case "TL":
		col, row = 0, 0
	case "TR":
		col, row = 2, 0
	case "BL":
		col, row = 0, 2
	case "BR":
		col, row = 2, 2
	default:
		return 0, 0
	}
	cx := (xBounds[col][0] + xBounds[col][1]) / 2
	cy := (yBounds[row][0] + yBounds[row][1]) / 2
	return cx, cy
}

// drawText draws a string at (x, y) using the given font face.
func drawText(img *image.RGBA, face font.Face, s string, x, y int, c color.RGBA) {
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c),
		Face: face,
		Dot:  fixed.P(x, y),
	}
	d.DrawString(s)
}

// drawCenteredText draws a string centered at (cx, cy).
func drawCenteredText(img *image.RGBA, face font.Face, s string, cx, cy int, c color.RGBA) {
	advance := font.MeasureString(face, s)
	x := cx - advance.Round()/2
	drawText(img, face, s, x, cy, c)
}

// drawCircle draws a circle outline using midpoint algorithm.
func drawCircle(img *image.RGBA, cx, cy, radius int, c color.RGBA) {
	x := radius
	y := 0
	err := 0

	for x >= y {
		img.Set(cx+x, cy+y, c)
		img.Set(cx+y, cy+x, c)
		img.Set(cx-y, cy+x, c)
		img.Set(cx-x, cy+y, c)
		img.Set(cx-x, cy-y, c)
		img.Set(cx-y, cy-x, c)
		img.Set(cx+y, cy-x, c)
		img.Set(cx+x, cy-y, c)

		y++
		err += 1 + 2*y
		if 2*(err-x)+1 > 0 {
			x--
			err += 1 - 2*x
		}
	}
}

// drawLine draws a line between two points using Bresenham's algorithm.
func drawLine(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	dx := intAbs(x1 - x0)
	dy := -intAbs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy

	for {
		img.Set(x0, y0, c)
		// Draw thicker line (3px)
		img.Set(x0+1, y0, c)
		img.Set(x0, y0+1, c)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

// drawArrowHead draws a simple arrowhead at the end of a line.
func drawArrowHead(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	dx := float64(x1 - x0)
	dy := float64(y1 - y0)
	length := math.Sqrt(dx*dx + dy*dy)
	if length < 1 {
		return
	}
	// Normalize
	dx /= length
	dy /= length

	// Arrow head size
	sz := 15.0
	// Two wing points perpendicular to the line direction
	px := float64(x1) - dx*sz
	py := float64(y1) - dy*sz

	wx1 := int(px + dy*sz*0.4)
	wy1 := int(py - dx*sz*0.4)
	wx2 := int(px - dy*sz*0.4)
	wy2 := int(py + dx*sz*0.4)

	drawLine(img, x1, y1, wx1, wy1, c)
	drawLine(img, x1, y1, wx2, wy2, c)
}

func intAbs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
