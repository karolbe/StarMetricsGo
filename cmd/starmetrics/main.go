package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	sm "starmetrics/pkg/starmetrics"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: starmetrics <input-file>")
	}
	inputFilePath := args[0]
	fmt.Printf("Loading: %s\n", inputFilePath)

	startTime := time.Now()
	imageWidth, imageHeight, stars, err := detectStars(inputFilePath)
	if err != nil {
		return err
	}
	elapsed := time.Since(startTime)

	starsWithPSF := make([]*sm.Star, 0)
	for _, s := range stars {
		if s.PSF != nil {
			starsWithPSF = append(starsWithPSF, s)
		}
	}

	fmt.Println()
	fmt.Printf("=== Star Detection Results (%.1fs) ===\n", elapsed.Seconds())
	fmt.Printf("  Image size:      %d x %d\n", imageWidth, imageHeight)
	fmt.Printf("  Stars detected:  %d\n", len(stars))
	fmt.Printf("  Stars with PSF:  %d\n", len(starsWithPSF))

	if len(stars) > 0 {
		hfrValues := make([]float64, len(stars))
		for i, s := range stars {
			hfrValues[i] = s.HFR
		}
		hfrMedian, hfrMAD := medianMAD(hfrValues)
		fmt.Printf("  HFR (median):    %.3f +/- %.3f px\n", hfrMedian, hfrMAD)
	}

	if len(starsWithPSF) > 0 {
		fwhmPxValues := make([]float64, len(starsWithPSF))
		fwhmAsValues := make([]float64, len(starsWithPSF))
		eccValues := make([]float64, len(starsWithPSF))
		for i, s := range starsWithPSF {
			fwhmPxValues[i] = (s.PSF.FWHMx + s.PSF.FWHMy) / 2.0
			fwhmAsValues[i] = s.PSF.FWHMArcsecs
			eccValues[i] = s.PSF.Eccentricity
		}
		fwhmPxMedian, fwhmPxMAD := medianMAD(fwhmPxValues)
		fwhmMedian, fwhmMAD := medianMAD(fwhmAsValues)
		eccMedian, eccMAD := medianMAD(eccValues)

		fmt.Printf("  FWHM (median):   %.3f +/- %.3f px\n", fwhmPxMedian, fwhmPxMAD)
		fmt.Printf("  FWHM (arcsec):   %.3f +/- %.3f\"\n", fwhmMedian, fwhmMAD)
		fmt.Printf("  Eccentricity:    %.3f +/- %.3f\n", eccMedian, eccMAD)
	}
	fmt.Println("==============================")

	// Field analysis (tilt)
	field := sm.AnalyzeField(stars, imageWidth, imageHeight)
	if field != nil {
		fmt.Println()
		fmt.Println("=== Field Analysis (3x3) ===")
		zoneOrder := []sm.ZonePosition{
			sm.ZoneTopLeft, sm.ZoneTop, sm.ZoneTopRight,
			sm.ZoneLeft, sm.ZoneCenter, sm.ZoneRight,
			sm.ZoneBottomLeft, sm.ZoneBottom, sm.ZoneBottomRight,
		}
		for i, pos := range zoneOrder {
			z := field.Zones[pos]
			fmt.Printf("  %-8s HFR=%.3f  FWHM=%.3f  n=%d\n", z.Label, z.MedianHFR, z.MedianFWHM, z.StarCount)
			if (i+1)%3 == 0 && i < 8 {
				fmt.Println("  ---")
			}
		}
		fmt.Printf("\n  Tilt:     %.1f%% (best: %s, worst: %s)\n", field.TiltPct, field.BestCorner, field.WorstCorner)
		fmt.Printf("  Off-axis: %.1f%%\n", field.OffAxisPct)
		if !field.Reliable {
			fmt.Println("  [LOW STAR COUNT - UNRELIABLE]")
		}
		fmt.Println("==============================")
	}

	return nil
}

func detectStars(inputFilePath string) (int, int, []*sm.Star, error) {
	var srcFloat sm.Mat
	var imageWidth, imageHeight int

	lowerPath := strings.ToLower(inputFilePath)
	if strings.HasSuffix(lowerPath, ".fits") || strings.HasSuffix(lowerPath, ".fit") {
		fitsData, err := sm.ReadFits(inputFilePath)
		if err != nil {
			return 0, 0, nil, fmt.Errorf("reading FITS: %w", err)
		}
		fmt.Printf("FITS loaded: %dx%d, %d-bit\n", fitsData.Width, fitsData.Height, fitsData.BitDepth)
		srcFloat = sm.ToFloat32Mat(fitsData.Pixels, fitsData.BitDepth, fitsData.Width, fitsData.Height)
		imageWidth = fitsData.Width
		imageHeight = fitsData.Height
	} else {
		var err error
		srcFloat, imageWidth, imageHeight, err = loadNonFitsImage(inputFilePath)
		if err != nil {
			return 0, 0, nil, err
		}
	}
	defer srcFloat.Close()

	const psfResolution = 10
	const psfGoodnessThreshold = 0.9
	const pixelScale = 1.0

	detectorParams := &sm.StarDetectorParams{
		HotpixelFiltering:                    true,
		HotpixelThresholdingEnabled:          true,
		HotpixelThreshold:                    0.001,
		StarMeasurementNoiseReductionEnabled: true,
		NoiseReductionRadius:                 3,
		NoiseClippingMultiplier:              4,
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
		PSFResolution:                        psfResolution,
		PSFGoodnessOfFitThreshold:            psfGoodnessThreshold,
		PixelScale:                           pixelScale,
	}

	result, err := sm.Detect(srcFloat, detectorParams, context.Background())
	if err != nil {
		return 0, 0, nil, fmt.Errorf("detecting stars: %w", err)
	}
	stars := result.DetectedStars

	// Fit PSFs in parallel
	fmt.Printf("Fitting PSFs for %d stars...\n", len(stars))
	psfStart := time.Now()
	var wg sync.WaitGroup
	for _, star := range stars {
		wg.Add(1)
		go func(s *sm.Star) {
			defer wg.Done()
			s.PSF = sm.FitStar(s, srcFloat, psfResolution, pixelScale, psfGoodnessThreshold)
		}(star)
	}
	wg.Wait()
	fmt.Printf("PSF fitting: %.1fs\n", time.Since(psfStart).Seconds())

	return imageWidth, imageHeight, stars, nil
}

func medianMAD(values []float64) (float64, float64) {
	if len(values) == 0 {
		return math.NaN(), math.NaN()
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	n := len(sorted)
	var median float64
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2.0
	} else {
		median = sorted[n/2]
	}

	deviations := make([]float64, n)
	for i := range sorted {
		deviations[i] = math.Abs(sorted[i] - median)
	}
	sort.Float64s(deviations)

	var madMedian float64
	if n%2 == 0 {
		madMedian = (deviations[n/2-1] + deviations[n/2]) / 2.0
	} else {
		madMedian = deviations[n/2]
	}

	return median, 1.4826 * madMedian
}
