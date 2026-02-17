/*
Extracted from HocusFocus plugin by George Hilios.
Original Copyright © 2021 George Hilios <ghilios+NINA@googlemail.com>
Licensed under Mozilla Public License 2.0.
Ported to Go.
*/

package starmetrics

import (
	"context"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"sort"
)

type starCandidate struct {
	Center              Point2d
	CenterBrightness    float64
	Background          float64
	NormalizedBrightness float64
	TotalFlux           float64
	Peak                float64
	PixelCount          int
	StarBoundingBox     image.Rectangle
	StarMedian          float64
}

// Detect runs the full star detection pipeline.
func Detect(srcImage Mat, p *StarDetectorParams, ctx context.Context) (*StarDetectorResult, error) {
	return detectImpl(srcImage, p, false, ctx)
}

func detectImpl(srcImage Mat, p *StarDetectorParams, hotpixelFilterAlreadyApplied bool, ctx context.Context) (*StarDetectorResult, error) {
	if p.HotpixelFiltering && p.HotpixelFilterRadius != 1 {
		return nil, fmt.Errorf("only hotpixel filter radius of 1 currently supported")
	}

	maybeSaveText(p.SaveIntermediateFilesPath, "00-params.txt",
		fmt.Sprintf("Params: NoiseReductionRadius=%d, NoiseClippingMultiplier=%f, StructureLayers=%d",
			p.NoiseReductionRadius, p.NoiseClippingMultiplier, p.StructureLayers))

	metrics := NewStarDetectorMetrics()
	debugData := &DebugData{}

	// Handle ROI
	var roiRect *image.Rectangle
	if p.Region.OuterBoundary.Height < 1.0 || p.Region.OuterBoundary.Width < 1.0 {
		r := image.Rect(
			int(math.Floor(float64(srcImage.Cols())*p.Region.OuterBoundary.StartX)),
			int(math.Floor(float64(srcImage.Rows())*p.Region.OuterBoundary.StartY)),
			int(math.Floor(float64(srcImage.Cols())*p.Region.OuterBoundary.StartX))+int(float64(srcImage.Cols())*p.Region.OuterBoundary.Width),
			int(math.Floor(float64(srcImage.Rows())*p.Region.OuterBoundary.StartY))+int(float64(srcImage.Rows())*p.Region.OuterBoundary.Height),
		)
		roiRect = &r
		debugData.DetectionROI = r

		roiView := srcImage.Region(r)
		roiClone := roiView.Clone()
		roiView.Close()
		srcImage.Close()
		srcImage = roiClone
		defer srcImage.Close()
	} else {
		debugData.DetectionROI = image.Rect(0, 0, srcImage.Cols(), srcImage.Rows())
	}
	maybeSaveImage(srcImage, p.SaveIntermediateFilesPath, "01-source.tif")

	if p.StoreStructureMap {
		debugData.StructureMap = make([]byte, debugData.DetectionROI.Dx()*debugData.DetectionROI.Dy())
	}

	// Step 1: Hotpixel filtering + noise reduction
	hotpixelFilteringApplied := hotpixelFilterAlreadyApplied
	if p.HotpixelFiltering || (p.NoiseReductionRadius > 0 && p.StarMeasurementNoiseReductionEnabled) {
		if !hotpixelFilterAlreadyApplied {
			metrics.HotpixelCount = applyHotpixelFilter(&srcImage, p)
		}
		hotpixelFilteringApplied = true
	}

	noiseReductionApplied := false
	if p.NoiseReductionRadius > 0 && p.StarMeasurementNoiseReductionEnabled {
		ConvolveGaussian(&srcImage, &srcImage, p.NoiseReductionRadius*2+1)
		noiseReductionApplied = true
	}
	maybeSaveImage(srcImage, p.SaveIntermediateFilesPath, "02-src-image-preparation.tif")

	// Step 2: Prepare noise-reduced image for structure detection
	noiseReducedImage := NewMat()
	defer noiseReducedImage.Close()
	CopyMatTo(srcImage, &noiseReducedImage)
	if !hotpixelFilteringApplied && !noiseReductionApplied {
		metrics.HotpixelCount = applyHotpixelFilter(&noiseReducedImage, p)
	}

	// Step 3: Apply noise reduction if not yet applied
	if p.NoiseReductionRadius > 0 && !noiseReductionApplied {
		ConvolveGaussian(&noiseReducedImage, &noiseReducedImage, p.NoiseReductionRadius*2+1)
	}

	structureMap := NewMat()
	defer structureMap.Close()
	CopyMatTo(noiseReducedImage, &structureMap)

	noiseReducedImageNoise := KappaSigmaNoiseEstimate(noiseReducedImage, p.NoiseClippingMultiplier, 0.00001, 5)
	maybeSaveText(p.SaveIntermediateFilesPath, "02-ksigma-estimate-noise-reduced.txt",
		fmt.Sprintf("K-Sigma Noise Estimate: %f, Background Mean: %f, NumIterations=%d",
			noiseReducedImageNoise.Sigma, noiseReducedImageNoise.BackgroundMean, noiseReducedImageNoise.NumIterations))

	maybeSaveImage(structureMap, p.SaveIntermediateFilesPath, "03-structure-map-start.tif")

	// Step 4: Compute B-spline wavelets to exclude large structures
	residualLayer := ComputeResidualAtrousB3SplineDyadicWaveletLayer(structureMap, p.StructureLayers)
	maybeSaveImage(residualLayer, p.SaveIntermediateFilesPath, "04-structure-wavelet-residual.tif")
	SubtractInPlace(&structureMap, &residualLayer)
	residualLayer.Close()

	maybeSaveImage(structureMap, p.SaveIntermediateFilesPath, "04-structure-wavelet-subtracted.tif")

	// Step 5: Post-wavelet blur
	ConvolveGaussian(&structureMap, &structureMap, p.StructureLayers*2+1)
	maybeSaveImage(structureMap, p.SaveIntermediateFilesPath, "05-structure-wavelet-blurred.tif")

	structureMapStats := CalculateStatisticsHistogram(structureMap, false, nil, nil, StatMedian)

	binarizeThreshold := structureMapStats.Median + p.NoiseClippingMultiplier*noiseReducedImageNoise.Sigma
	maybeSaveText(p.SaveIntermediateFilesPath, "05-structure-map-statistics.txt",
		fmt.Sprintf("%s\nBinarization - Median: %f, Threshold: %f, Clipping: %f, Noise Sigma: %f",
			structureMapStats.String(), structureMapStats.Median, binarizeThreshold,
			p.NoiseClippingMultiplier, noiseReducedImageNoise.Sigma))

	if p.StoreStructureMap {
		updateStructureMapDebugData(structureMap, debugData.StructureMap, binarizeThreshold, 1)
	}

	// Step 6: Dilation for small structures
	if p.StructureDilationCount > 0 {
		morphDilateEllipse(structureMap, &structureMap, p.StructureDilationSize, p.StructureDilationCount)
		maybeSaveImage(structureMap, p.SaveIntermediateFilesPath, "06-structure-map-dilated.tif")
	}

	if p.StoreStructureMap {
		updateStructureMapDebugData(structureMap, debugData.StructureMap, binarizeThreshold, 2)
	}

	// Step 7: Binarize
	Binarize(&structureMap, &structureMap, binarizeThreshold)
	if p.Region.InnerCropBoundary != nil {
		icb := p.Region.InnerCropBoundary
		innerROI := image.Rect(
			int(math.Floor(float64(srcImage.Cols())*icb.StartX)),
			int(math.Floor(float64(srcImage.Rows())*icb.StartY)),
			int(math.Floor(float64(srcImage.Cols())*icb.StartX))+int(float64(srcImage.Cols())*icb.Width),
			int(math.Floor(float64(srcImage.Rows())*icb.StartY))+int(float64(srcImage.Rows())*icb.Height),
		)
		innerRegion := structureMap.Region(innerROI)
		innerRegion.SetToZero()
		innerRegion.Close()
	}
	maybeSaveImage(structureMap, p.SaveIntermediateFilesPath, "07-structure-binarized.tif")

	// Step 8: Scan for stars
	stars := scanStars(srcImage, structureMap, p, noiseReducedImageNoise.Sigma, metrics, ctx)

	if roiRect != nil {
		for i, s := range stars {
			stars[i] = s.AddOffset(roiRect.Min.X, roiRect.Min.Y)
		}
		metrics.AddROIOffset(roiRect.Min.X, roiRect.Min.Y)
	}

	return &StarDetectorResult{
		DetectedStars: stars,
		Metrics:       metrics,
		DebugData:     debugData,
	}, nil
}

func applyHotpixelFilter(img *Mat, p *StarDetectorParams) int64 {
	if p.HotpixelThresholdingEnabled {
		return hotpixelFilterWithThresholding(img, p.HotpixelThreshold)
	}
	medianBlur(*img, img, 3)
	return 0
}

func hotpixelFilterWithThresholding(m *Mat, threshold float64) int64 {
	blurred := NewMat()
	defer blurred.Close()
	diff := NewMat()
	defer diff.Close()
	mask := NewMat()
	defer mask.Close()

	medianBlur(*m, &blurred, 3)
	absDiff(*m, blurred, &diff)
	thresholdBinary(diff, &mask, float32(threshold), 1.0)
	numHotpixels := int64(countNonZero(mask))
	matCopyToWithMask(blurred, m, mask)
	return numHotpixels
}

func updateStructureMapDebugData(structureMap Mat, debugData []byte, binarizationThreshold float64, value byte) {
	data := structureMap.DataFloat32()
	numPixels := structureMap.Rows() * structureMap.Cols()
	for i := 0; i < numPixels; i++ {
		if float64(data[i]) >= binarizationThreshold && debugData[i] == 0 {
			debugData[i] = value
		}
	}
}

func measureStar(srcImage Mat, star *Star, p *StarDetectorParams, noiseSigma float64) bool {
	background := star.Background
	totalBrightness := 0.0
	totalWeightedDistance := 0.0

	bb := star.StarBoundingBox
	bbLeft := float64(bb.Min.X)
	bbTop := float64(bb.Min.Y)
	bbRight := float64(bb.Max.X)
	bbBottom := float64(bb.Max.Y)

	startX := star.Center.X - float64(p.AnalysisSamplingSize)*math.Floor((star.Center.X-bbLeft)/float64(p.AnalysisSamplingSize))
	startY := star.Center.Y - float64(p.AnalysisSamplingSize)*math.Floor((star.Center.Y-bbTop)/float64(p.AnalysisSamplingSize))
	noiseThreshold := p.StarClippingMultiplier * noiseSigma

	for y := startY; y <= bbBottom; y += float64(p.AnalysisSamplingSize) {
		for x := startX; x <= bbRight; x += float64(p.AnalysisSamplingSize) {
			value := BilinearSamplePixelValue(srcImage, y, x) - background - noiseThreshold
			if value > 0.0 {
				dx := x - star.Center.X
				dy := y - star.Center.Y
				distance := math.Sqrt(dx*dx + dy*dy)
				totalWeightedDistance += value * distance
				totalBrightness += value
			}
		}
	}

	if totalBrightness > 0.0 {
		star.HFR = totalWeightedDistance / totalBrightness
		return true
	}
	return false
}

func evaluateGlobalMetrics(srcImage Mat, p *StarDetectorParams, metrics *StarDetectorMetrics) {
	data := srcImage.DataFloat32()
	numPixels := srcImage.Rows() * srcImage.Cols()
	thresh := float32(p.SaturationThreshold)
	for i := 0; i < numPixels; i++ {
		if data[i] >= thresh {
			metrics.SaturatedPixelCount++
		}
	}
}

func scanStars(srcImage, structureMap Mat, p *StarDetectorParams, srcImageNoiseSigma float64, metrics *StarDetectorMetrics, ctx context.Context) []*Star {
	const zeroThreshold float32 = 0.001

	stars := make([]*Star, 0)
	starPoints := make([]image.Point, 0, 1024)
	width := structureMap.Cols()
	height := structureMap.Rows()
	evaluateGlobalMetrics(srcImage, p, metrics)

	structureData := structureMap.DataFloat32()
	xRight := width - 1
	yBottom := height - 1

	for yTop := 0; yTop < yBottom; yTop++ {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return stars
			default:
			}
		}

		for xLeft := 0; xLeft < xRight; xLeft++ {
			if structureData[yTop*width+xLeft] < zeroThreshold {
				continue
			}

			starPoints = starPoints[:0]
			starBounds := image.Rect(xLeft, yTop, xLeft+1, yTop+1)

			y := yTop
			x := xLeft
			for {
				rowOffsetStart := y * width
				rowPointsAdded := 0
				if structureData[rowOffsetStart+x] >= zeroThreshold {
					starPoints = append(starPoints, image.Pt(x, y))
					rowPointsAdded++
				}

				rowStartX := x
				rowEndX := x

				if rowPointsAdded > 0 {
					for rowStartX > 0 {
						if structureData[rowOffsetStart+(rowStartX-1)] < zeroThreshold {
							break
						}
						rowStartX--
						starPoints = append(starPoints, image.Pt(rowStartX, y))
						rowPointsAdded++
					}
				}

				for rowEndX < xRight {
					if structureData[rowOffsetStart+(rowEndX+1)] < zeroThreshold {
						if rowPointsAdded > 0 || rowEndX >= starBounds.Max.X {
							break
						}
						rowEndX++
					} else {
						rowEndX++
						starPoints = append(starPoints, image.Pt(rowEndX, y))
						rowPointsAdded++
					}
				}

				if rowStartX < starBounds.Min.X {
					starBounds.Max.X += starBounds.Min.X - rowStartX
					starBounds.Min.X = rowStartX
				}
				if rowEndX > starBounds.Max.X-1 {
					starBounds.Max.X = rowEndX + 1
				}

				if rowPointsAdded == 0 {
					starBounds.Max.Y = y
					break
				}
				if y == yBottom {
					starBounds.Max.Y = y + 1
					break
				}

				y++
			}

			metrics.StructureCandidates++
			star := evaluateStarCandidate(srcImage, p, starBounds, starPoints, srcImageNoiseSigma, metrics)
			if star != nil {
				metrics.TotalDetected++
				stars = append(stars, star)
			}

			// Clamp bounds to image dimensions (Max.X can exceed width
			// due to symmetric search window expansion on line 331)
			clampedMaxX := starBounds.Max.X
			if clampedMaxX > width {
				clampedMaxX = width
			}
			clampedMaxY := starBounds.Max.Y
			if clampedMaxY > height {
				clampedMaxY = height
			}
			for cy := starBounds.Min.Y; cy < clampedMaxY; cy++ {
				for cx := starBounds.Min.X; cx < clampedMaxX; cx++ {
					structureData[cy*width+cx] = 0.0
				}
			}
		}
	}

	return stars
}

func evaluateStarCandidate(srcImage Mat, p *StarDetectorParams, starBounds image.Rectangle, starPoints []image.Point, srcImageNoiseSigma float64, metrics *StarDetectorMetrics) *Star {
	bbWidth := starBounds.Dx()
	bbHeight := starBounds.Dy()

	if bbWidth < p.MinimumStarBoundingBoxSize || bbHeight < p.MinimumStarBoundingBoxSize {
		metrics.TooSmall++
		return nil
	}

	if starBounds.Min.X == 0 || starBounds.Min.Y == 0 || starBounds.Max.X >= srcImage.Cols() || starBounds.Max.Y >= srcImage.Rows() {
		metrics.OnBorder++
		return nil
	}

	d := math.Max(float64(bbWidth), float64(bbHeight))
	if float64(len(starPoints))/(d*d) < p.MaxDistortion {
		metrics.TooDistortedBounds = append(metrics.TooDistortedBounds, starBounds)
		return nil
	}

	candidate := computeStarParameters(srcImage, starBounds, p, srcImageNoiseSigma, starPoints)
	if candidate == nil {
		metrics.DegenerateBounds = append(metrics.DegenerateBounds, starBounds)
		return nil
	}

	if (candidate.Background + candidate.Peak) >= p.SaturationThreshold {
		metrics.SaturatedBounds = append(metrics.SaturatedBounds, starBounds)
		return nil
	}

	sensitivity := candidate.NormalizedBrightness / srcImageNoiseSigma
	if sensitivity <= p.Sensitivity {
		metrics.LowSensitivityBounds = append(metrics.LowSensitivityBounds, starBounds)
		return nil
	}

	if !isStarCentered(candidate, p) {
		metrics.NotCenteredBounds = append(metrics.NotCenteredBounds, starBounds)
		return nil
	}

	if candidate.StarMedian >= (p.PeakResponse * candidate.Peak) {
		metrics.TooFlatBounds = append(metrics.TooFlatBounds, starBounds)
		return nil
	}

	star := &Star{
		Center:         candidate.Center,
		Background:     candidate.Background,
		MeanBrightness: candidate.TotalFlux / float64(candidate.PixelCount),
		PeakBrightness: candidate.Peak,
		Flux:           candidate.TotalFlux,
		StarBoundingBox: starBounds,
	}

	if !measureStar(srcImage, star, p, srcImageNoiseSigma) {
		metrics.HFRAnalysisFailed++
		return nil
	}

	if star.HFR <= p.MinHFR {
		metrics.TooLowHFR++
		return nil
	}

	return star
}

func isStarCentered(candidate *starCandidate, p *StarDetectorParams) bool {
	box := candidate.StarBoundingBox
	ctw := float64(box.Dx()) * p.StarCenterTolerance
	cth := float64(box.Dy()) * p.StarCenterTolerance
	minX := float64(box.Min.X) + (float64(box.Dx())-ctw)/2.0
	maxX := minX + ctw
	minY := float64(box.Min.Y) + (float64(box.Dy())-cth)/2.0
	maxY := minY + cth
	return candidate.Center.X >= minX && candidate.Center.X <= maxX &&
		candidate.Center.Y >= minY && candidate.Center.Y <= maxY
}

func computeStarParameters(srcImage Mat, starBounds image.Rectangle, p *StarDetectorParams, noiseSigma float64, starPoints []image.Point) *starCandidate {
	imgWidth := srcImage.Cols()
	imgHeight := srcImage.Rows()
	imageData := srcImage.DataFloat32()

	backgroundStartY := starBounds.Min.Y - p.BackgroundBoxExpansion
	if backgroundStartY < 0 {
		backgroundStartY = 0
	}
	backgroundStartX := starBounds.Min.X - p.BackgroundBoxExpansion
	if backgroundStartX < 0 {
		backgroundStartX = 0
	}
	backgroundEndY := starBounds.Max.Y + p.BackgroundBoxExpansion
	if backgroundEndY > imgHeight {
		backgroundEndY = imgHeight
	}
	backgroundEndX := starBounds.Max.X + p.BackgroundBoxExpansion
	if backgroundEndX > imgWidth {
		backgroundEndX = imgWidth
	}

	surroundingPixels := make([]float32, 0, (backgroundEndX-backgroundStartX)*(backgroundEndY-backgroundStartY))

	for y := backgroundStartY; y < starBounds.Min.Y; y++ {
		for x := backgroundStartX; x < backgroundEndX; x++ {
			surroundingPixels = append(surroundingPixels, imageData[y*imgWidth+x])
		}
	}
	for y := starBounds.Min.Y; y < starBounds.Max.Y; y++ {
		for x := backgroundStartX; x < starBounds.Min.X; x++ {
			surroundingPixels = append(surroundingPixels, imageData[y*imgWidth+x])
		}
		for x := starBounds.Max.X; x < backgroundEndX; x++ {
			surroundingPixels = append(surroundingPixels, imageData[y*imgWidth+x])
		}
	}
	for y := starBounds.Max.Y; y < backgroundEndY; y++ {
		for x := backgroundStartX; x < backgroundEndX; x++ {
			surroundingPixels = append(surroundingPixels, imageData[y*imgWidth+x])
		}
	}

	sort.Slice(surroundingPixels, func(i, j int) bool { return surroundingPixels[i] < surroundingPixels[j] })
	backgroundMedian := surroundingPixels[len(surroundingPixels)/2]

	backgroundThreshold := float64(backgroundMedian) + p.StarClippingMultiplier*noiseSigma
	var sx, sy, sz, totalFlux, peak float64
	numUnclippedPixels := 0
	var minPixel float32 = 1.0
	var maxPixel float32 = 0.0

	for _, sp := range starPoints {
		pixel := imageData[sp.Y*imgWidth+sp.X]
		if float64(pixel) <= backgroundThreshold {
			continue
		}
		pixel -= backgroundMedian
		numUnclippedPixels++
		if pixel < minPixel {
			minPixel = pixel
		}
		if pixel > maxPixel {
			maxPixel = pixel
		}
	}

	if numUnclippedPixels <= 1 || maxPixel <= minPixel {
		return nil
	}

	starPixels := make([]float64, 0, numUnclippedPixels)
	for _, sp := range starPoints {
		pixel := imageData[sp.Y*imgWidth+sp.X]
		if float64(pixel) <= backgroundThreshold {
			continue
		}
		pixel -= backgroundMedian
		fPixel := float64(pixel)
		sx += fPixel * float64(sp.X)
		sy += fPixel * float64(sp.Y)
		sz += fPixel
		totalFlux += fPixel
		if fPixel > peak {
			peak = fPixel
		}
		starPixels = append(starPixels, fPixel)
	}

	sort.Float64s(starPixels)
	var starMedian float64
	n := len(starPixels)
	if n%2 == 1 {
		starMedian = starPixels[n/2]
	} else {
		starMedian = (starPixels[n/2] + starPixels[n/2-1]) / 2.0
	}

	meanFlux := totalFlux / float64(len(starPoints))
	center := Point2d{X: sx / sz, Y: sy / sz}
	centerBrightness := BilinearSamplePixelValue(srcImage, center.Y, center.X) - float64(backgroundMedian)

	return &starCandidate{
		Center:              center,
		CenterBrightness:    centerBrightness,
		Background:          float64(backgroundMedian),
		TotalFlux:           totalFlux,
		Peak:                peak,
		NormalizedBrightness: peak - (1-p.PeakResponse)*meanFlux,
		StarBoundingBox:     starBounds,
		StarMedian:          starMedian,
		PixelCount:          len(starPoints),
	}
}

func maybeSaveImage(img Mat, savePath, filename string) {
	if savePath == "" {
		return
	}
	if _, err := os.Stat(savePath); os.IsNotExist(err) {
		return
	}
	imWriteMat(filepath.Join(savePath, filename), img)
}

func maybeSaveText(savePath, filename, text string) {
	if savePath == "" {
		return
	}
	if _, err := os.Stat(savePath); os.IsNotExist(err) {
		return
	}
	os.WriteFile(filepath.Join(savePath, filename), []byte(text), 0644)
}
