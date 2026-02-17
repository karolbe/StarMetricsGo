package starmetrics

import (
	"math"
	"sort"
)

const (
	fieldEdgeFraction    = 0.25
	minStarsPerZone      = 3
	minTotalStarsForTilt = 20
)

var zoneLabels = map[ZonePosition]string{
	ZoneTopLeft:     "TL",
	ZoneTop:         "T",
	ZoneTopRight:    "TR",
	ZoneLeft:        "L",
	ZoneCenter:      "Center",
	ZoneRight:       "R",
	ZoneBottomLeft:  "BL",
	ZoneBottom:      "B",
	ZoneBottomRight: "BR",
}

var cornerPositions = []ZonePosition{ZoneTopLeft, ZoneTopRight, ZoneBottomLeft, ZoneBottomRight}

// AnalyzeField divides the image into a 3x3 grid and computes per-zone
// HFR/FWHM statistics, tilt metrics, and best/worst corners.
func AnalyzeField(stars []*Star, width, height int) *FieldAnalysis {
	if len(stars) == 0 {
		return nil
	}

	xLo := float64(width) * fieldEdgeFraction
	xHi := float64(width) * (1.0 - fieldEdgeFraction)
	yLo := float64(height) * fieldEdgeFraction
	yHi := float64(height) * (1.0 - fieldEdgeFraction)

	// Bucket stars into zones
	zoneStars := make(map[ZonePosition][]*Star)
	for _, pos := range []ZonePosition{
		ZoneTopLeft, ZoneTop, ZoneTopRight,
		ZoneLeft, ZoneCenter, ZoneRight,
		ZoneBottomLeft, ZoneBottom, ZoneBottomRight,
	} {
		zoneStars[pos] = make([]*Star, 0)
	}

	for _, s := range stars {
		pos := classifyZone(s.Center.X, s.Center.Y, xLo, xHi, yLo, yHi)
		zoneStars[pos] = append(zoneStars[pos], s)
	}

	// Compute per-zone statistics
	zones := make(map[ZonePosition]ZoneData)
	for pos, starList := range zoneStars {
		zones[pos] = computeZoneData(pos, starList)
	}

	result := &FieldAnalysis{
		Zones: zones,
	}

	centerHFR := zones[ZoneCenter].MedianHFR
	if centerHFR <= 0 {
		result.Reliable = false
		return result
	}

	// Tilt: compare corners to center
	var bestCorner, worstCorner ZonePosition
	bestHFR := math.MaxFloat64
	worstHFR := 0.0
	validCorners := 0

	for _, pos := range cornerPositions {
		z := zones[pos]
		if z.StarCount < minStarsPerZone {
			continue
		}
		validCorners++
		if z.MedianHFR < bestHFR {
			bestHFR = z.MedianHFR
			bestCorner = pos
		}
		if z.MedianHFR > worstHFR {
			worstHFR = z.MedianHFR
			worstCorner = pos
		}
	}

	if validCorners >= 2 && worstHFR > 0 {
		result.TiltPct = (worstHFR - bestHFR) / centerHFR * 100.0
		result.BestCorner = zoneLabels[bestCorner]
		result.WorstCorner = zoneLabels[worstCorner]
	}

	// Off-axis: average of all non-center zones vs center
	var offAxisSum float64
	offAxisCount := 0
	for pos, z := range zones {
		if pos == ZoneCenter || z.StarCount < minStarsPerZone {
			continue
		}
		offAxisSum += z.MedianHFR
		offAxisCount++
	}
	if offAxisCount > 0 {
		avgOffAxis := offAxisSum / float64(offAxisCount)
		result.OffAxisPct = (avgOffAxis - centerHFR) / centerHFR * 100.0
	}

	result.Reliable = len(stars) >= minTotalStarsForTilt && validCorners >= 4 && zones[ZoneCenter].StarCount >= minStarsPerZone

	return result
}

func classifyZone(x, y, xLo, xHi, yLo, yHi float64) ZonePosition {
	var col, row int
	if x < xLo {
		col = 0
	} else if x < xHi {
		col = 1
	} else {
		col = 2
	}
	if y < yLo {
		row = 0
	} else if y < yHi {
		row = 1
	} else {
		row = 2
	}

	grid := [3][3]ZonePosition{
		{ZoneTopLeft, ZoneTop, ZoneTopRight},
		{ZoneLeft, ZoneCenter, ZoneRight},
		{ZoneBottomLeft, ZoneBottom, ZoneBottomRight},
	}
	return grid[row][col]
}

func computeZoneData(pos ZonePosition, stars []*Star) ZoneData {
	zd := ZoneData{
		Label:     zoneLabels[pos],
		StarCount: len(stars),
	}
	if len(stars) == 0 {
		return zd
	}

	hfrValues := make([]float64, len(stars))
	for i, s := range stars {
		hfrValues[i] = s.HFR
	}
	zd.MedianHFR = medianFloat64(hfrValues)

	fwhmValues := make([]float64, 0, len(stars))
	for _, s := range stars {
		if s.PSF != nil {
			fwhmValues = append(fwhmValues, s.PSF.FWHMPixels)
		}
	}
	if len(fwhmValues) > 0 {
		zd.MedianFWHM = medianFloat64(fwhmValues)
	}

	return zd
}

func medianFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2.0
	}
	return sorted[n/2]
}
