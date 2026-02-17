//go:build js && wasm

package main

import (
	"context"
	"math"
	"sort"
	"sync"
	"syscall/js"

	sm "starmetrics/pkg/starmetrics"
)

var (
	lastField  *sm.FieldAnalysis
	lastWidth  int
	lastHeight int
)

func main() {
	js.Global().Set("analyzeFITS", js.FuncOf(analyzeFITS))
	js.Global().Set("renderOverlay", js.FuncOf(renderOverlay))
	select {} // block forever
}

func analyzeFITS(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return errorResult("usage: analyzeFITS(fileBytes, sigma, options)")
	}

	// Extract file bytes
	jsBytes := args[0]
	length := jsBytes.Get("length").Int()
	fileBytes := make([]byte, length)
	js.CopyBytesToGo(fileBytes, jsBytes)

	sigma := args[1].Float()

	debayer := false
	if len(args) >= 3 && args[2].Type() == js.TypeObject {
		debayerVal := args[2].Get("debayer")
		if debayerVal.Type() == js.TypeBoolean {
			debayer = debayerVal.Bool()
		}
	}

	// Parse FITS
	fitsData, err := sm.ReadFitsFromBytes(fileBytes)
	if err != nil {
		return errorResult("FITS parse error: " + err.Error())
	}

	// Create Mat (optionally debayer)
	var srcFloat sm.Mat
	if debayer {
		srcFloat = sm.DebayerToMat(fitsData.Pixels, fitsData.BitDepth, fitsData.Width, fitsData.Height)
	} else {
		srcFloat = sm.ToFloat32Mat(fitsData.Pixels, fitsData.BitDepth, fitsData.Width, fitsData.Height)
	}
	defer srcFloat.Close()

	imageWidth := fitsData.Width
	imageHeight := fitsData.Height

	// Configure detector
	params := &sm.StarDetectorParams{
		HotpixelFiltering:                    true,
		HotpixelThresholdingEnabled:          true,
		HotpixelThreshold:                    0.001,
		StarMeasurementNoiseReductionEnabled: true,
		NoiseReductionRadius:                 3,
		NoiseClippingMultiplier:              sigma,
		StarClippingMultiplier:               2,
		HotpixelFilterRadius:                 1,
		StructureLayers:                      4,
		MinimumStarBoundingBoxSize:           5,
		AnalysisSamplingSize:                 1.0,
		Sensitivity:                          10.0,
		PeakResponse:                         0.75,
		MaxDistortion:                        0.5,
		StarCenterTolerance:                  0.3,
		MinHFR:                               1.5,
		StructureDilationCount:               0,
		StructureDilationSize:                3,
		BackgroundBoxExpansion:               3,
		SaturationThreshold:                  0.99,
		Region:                               sm.StarDetectionRegionFull,
		ModelPSF:                             false,
		PSFResolution:                        10,
		PSFGoodnessOfFitThreshold:            0.9,
		PixelScale:                           1.0,
	}

	// Detect stars
	result, err := sm.Detect(srcFloat, params, context.Background())
	if err != nil {
		return errorResult("Detection error: " + err.Error())
	}
	stars := result.DetectedStars

	// Fit PSFs in parallel
	const psfResolution = 10
	const psfGoodnessThreshold = 0.9
	const pixelScale = 1.0

	var wg sync.WaitGroup
	for _, star := range stars {
		wg.Add(1)
		go func(s *sm.Star) {
			defer wg.Done()
			s.PSF = sm.FitStar(s, srcFloat, psfResolution, pixelScale, psfGoodnessThreshold)
		}(star)
	}
	wg.Wait()

	// Compute noise estimate for background/stddev
	noiseEst := sm.KappaSigmaNoiseEstimate(srcFloat, 4.0, 0.00001, 5)

	// Compute statistics
	hfrValues := make([]float64, len(stars))
	for i, s := range stars {
		hfrValues[i] = s.HFR
	}
	medianHFR, meanHFR, stddevHFR := computeStats(hfrValues)

	fwhmValues := make([]float64, 0, len(stars))
	eccValues := make([]float64, 0, len(stars))
	for _, s := range stars {
		if s.PSF != nil {
			fwhmValues = append(fwhmValues, s.PSF.FWHMPixels)
			eccValues = append(eccValues, s.PSF.Eccentricity)
		}
	}
	medianFWHM := medianF64(fwhmValues)
	medianEcc := medianF64(eccValues)

	// Field analysis
	field := sm.AnalyzeField(stars, imageWidth, imageHeight)
	lastField = field
	lastWidth = imageWidth
	lastHeight = imageHeight

	// Build JS result
	jsResult := map[string]interface{}{
		"width":              imageWidth,
		"height":             imageHeight,
		"background":         noiseEst.BackgroundMean,
		"stddev":             noiseEst.Sigma,
		"medianHFR":          medianHFR,
		"meanHFR":            meanHFR,
		"stddevHFR":          stddevHFR,
		"medianFWHM":         medianFWHM,
		"medianEccentricity": medianEcc,
	}

	// Stars array
	jsStars := make([]interface{}, len(stars))
	for i, s := range stars {
		fwhm := 0.0
		ecc := 0.0
		roundness := 0.0
		if s.PSF != nil {
			fwhm = s.PSF.FWHMPixels
			ecc = s.PSF.Eccentricity
			minFW := math.Min(s.PSF.FWHMx, s.PSF.FWHMy)
			maxFW := math.Max(s.PSF.FWHMx, s.PSF.FWHMy)
			if maxFW > 0 {
				roundness = minFW / maxFW
			}
		}
		jsStars[i] = map[string]interface{}{
			"x":            s.Center.X,
			"y":            s.Center.Y,
			"peak":         s.PeakBrightness,
			"flux":         s.Flux,
			"hfr":          s.HFR,
			"fwhm":         fwhm,
			"roundness":    roundness,
			"eccentricity": ecc,
		}
	}
	jsResult["stars"] = jsStars

	// Field analysis
	if field != nil {
		zoneOrder := []sm.ZonePosition{
			sm.ZoneTopLeft, sm.ZoneTop, sm.ZoneTopRight,
			sm.ZoneLeft, sm.ZoneCenter, sm.ZoneRight,
			sm.ZoneBottomLeft, sm.ZoneBottom, sm.ZoneBottomRight,
		}
		jsZones := make([]interface{}, len(zoneOrder))
		for i, pos := range zoneOrder {
			z := field.Zones[pos]
			jsZones[i] = map[string]interface{}{
				"label":      z.Label,
				"medianHFR":  z.MedianHFR,
				"starCount":  z.StarCount,
				"medianFWHM": z.MedianFWHM,
			}
		}
		jsResult["field"] = map[string]interface{}{
			"zones":       jsZones,
			"tiltPct":     field.TiltPct,
			"offAxisPct":  field.OffAxisPct,
			"bestCorner":  field.BestCorner,
			"worstCorner": field.WorstCorner,
			"reliable":    field.Reliable,
		}
	}

	return js.ValueOf(jsResult)
}

func renderOverlay(this js.Value, args []js.Value) interface{} {
	if lastField == nil {
		return js.Null()
	}

	jpegBytes, err := sm.RenderFieldOverlayBytes(lastField, lastWidth, lastHeight)
	if err != nil {
		return js.Null()
	}

	// Create Uint8Array and copy bytes
	uint8Array := js.Global().Get("Uint8Array").New(len(jpegBytes))
	js.CopyBytesToJS(uint8Array, jpegBytes)
	return uint8Array
}

func errorResult(msg string) interface{} {
	return js.ValueOf(map[string]interface{}{
		"error": msg,
	})
}

func computeStats(values []float64) (median, mean, stddev float64) {
	if len(values) == 0 {
		return 0, 0, 0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	n := len(sorted)
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2.0
	} else {
		median = sorted[n/2]
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean = sum / float64(n)

	sse := 0.0
	for _, v := range values {
		d := v - mean
		sse += d * d
	}
	if n > 1 {
		stddev = math.Sqrt(sse / float64(n-1))
	}
	return
}

func medianF64(values []float64) float64 {
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
