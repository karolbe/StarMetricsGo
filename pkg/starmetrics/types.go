/*
Extracted from HocusFocus plugin by George Hilios.
Original Copyright © 2021 George Hilios <ghilios+NINA@googlemail.com>
Licensed under Mozilla Public License 2.0.
Ported to Go.
*/

package starmetrics

import (
	"fmt"
	"image"
	"math"
)

// StarDetectorPSFFitType represents PSF fitting model types.
type StarDetectorPSFFitType int

const (
	PSFFitMoffat40 StarDetectorPSFFitType = iota
	PSFFitGaussian
)

func (t StarDetectorPSFFitType) String() string {
	switch t {
	case PSFFitMoffat40:
		return "Moffat_40"
	case PSFFitGaussian:
		return "Gaussian"
	default:
		return "Unknown"
	}
}

// RatioRect represents a rectangle defined by ratios in [0, 1).
type RatioRect struct {
	StartX float64
	StartY float64
	Width  float64
	Height float64
}

// RatioRectFull is a RatioRect covering the entire image.
var RatioRectFull = RatioRect{StartX: 0, StartY: 0, Width: 1, Height: 1}

// NewRatioRect creates a new RatioRect with validation.
func NewRatioRect(startX, startY, width, height float64) (RatioRect, error) {
	if startX < 0 || startX >= 1 {
		return RatioRect{}, fmt.Errorf("startX must be in [0, 1), got %f", startX)
	}
	if startY < 0 || startY >= 1 {
		return RatioRect{}, fmt.Errorf("startY must be in [0, 1), got %f", startY)
	}
	if width <= 0 {
		return RatioRect{}, fmt.Errorf("width must be positive, got %f", width)
	}
	if height <= 0 {
		return RatioRect{}, fmt.Errorf("height must be positive, got %f", height)
	}
	return RatioRect{
		StartX: startX,
		StartY: startY,
		Width:  math.Min(width, 1.0-startX),
		Height: math.Min(height, 1.0-startY),
	}, nil
}

func (r RatioRect) EndExclusiveX() float64 { return r.StartX + r.Width }
func (r RatioRect) EndExclusiveY() float64 { return r.StartY + r.Height }

func (r RatioRect) Contains(inner RatioRect) bool {
	if r.StartX > inner.StartX || r.StartY > inner.StartY {
		return false
	}
	if r.EndExclusiveX() < inner.EndExclusiveX() || r.EndExclusiveY() < inner.EndExclusiveY() {
		return false
	}
	return true
}

func (r RatioRect) IsFull() bool {
	return r.Width >= 1 && r.Height >= 1
}

// RatioRectFromCenterROI creates a RatioRect centered on the image with the given ROI ratio.
func RatioRectFromCenterROI(roi float64) RatioRect {
	return RatioRect{
		StartX: (1.0 - roi) / 2.0,
		StartY: (1.0 - roi) / 2.0,
		Width:  roi,
		Height: roi,
	}
}

// StarDetectionRegion defines the detection region.
type StarDetectionRegion struct {
	OuterBoundary     RatioRect
	InnerCropBoundary *RatioRect
	Index             int
}

// StarDetectionRegionFull covers the entire image.
var StarDetectionRegionFull = StarDetectionRegion{OuterBoundary: RatioRectFull}

func (r StarDetectionRegion) IsFull() bool {
	return r.InnerCropBoundary == nil && r.OuterBoundary.IsFull()
}

// StarDetectorParams contains all parameters for star detection.
type StarDetectorParams struct {
	HotpixelFiltering                bool
	HotpixelThresholdingEnabled      bool
	HotpixelThreshold                float64
	StarMeasurementNoiseReductionEnabled bool
	NoiseReductionRadius             int
	NoiseClippingMultiplier          float64
	StarClippingMultiplier           float64
	HotpixelFilterRadius             int
	StructureLayers                  int
	StructureDilationSize            int
	StructureDilationCount           int
	Sensitivity                      float64
	PeakResponse                     float64
	MaxDistortion                    float64
	StarCenterTolerance              float64
	BackgroundBoxExpansion           int
	MinimumStarBoundingBoxSize       int
	MinHFR                           float64
	Region                           StarDetectionRegion
	AnalysisSamplingSize             float32
	StoreStructureMap                bool
	SaveIntermediateFilesPath        string
	SaturationThreshold              float64
	ModelPSF                         bool
	PSFFitType                       StarDetectorPSFFitType
	PSFGoodnessOfFitThreshold        float64
	PSFResolution                    int
	PSFParallelPartitionSize         int
	PixelScale                       float64
}

// NewStarDetectorParams creates a StarDetectorParams with default values.
func NewStarDetectorParams() *StarDetectorParams {
	return &StarDetectorParams{
		HotpixelFiltering:                true,
		HotpixelThresholdingEnabled:      true,
		HotpixelThreshold:                0.001,
		StarMeasurementNoiseReductionEnabled: true,
		NoiseReductionRadius:             3,
		NoiseClippingMultiplier:          4.0,
		StarClippingMultiplier:           2.0,
		HotpixelFilterRadius:             1,
		StructureLayers:                  4,
		StructureDilationSize:            3,
		StructureDilationCount:           0,
		Sensitivity:                      10.0,
		PeakResponse:                     0.75,
		MaxDistortion:                    0.5,
		StarCenterTolerance:              0.3,
		BackgroundBoxExpansion:           3,
		MinimumStarBoundingBoxSize:       5,
		MinHFR:                           1.5,
		Region:                           StarDetectionRegionFull,
		AnalysisSamplingSize:             1.0,
		StoreStructureMap:                false,
		SaturationThreshold:              0.99,
		ModelPSF:                         true,
		PSFFitType:                       PSFFitMoffat40,
		PSFGoodnessOfFitThreshold:        0.9,
		PSFResolution:                    10,
		PSFParallelPartitionSize:         100,
		PixelScale:                       1.0,
	}
}

// Point2d represents a 2D point with float64 coordinates.
type Point2d struct {
	X, Y float64
}

// Star represents a detected star.
type Star struct {
	Center         Point2d
	StarBoundingBox image.Rectangle
	Background     float64
	MeanBrightness float64
	PeakBrightness float64
	Flux           float64
	HFR            float64
	PSF            *PSFModel
}

func (s *Star) String() string {
	return fmt.Sprintf("{Center=(%f,%f), BBox=%v, Background=%f, MeanBrightness=%f, PeakBrightness=%f, Flux=%f, HFR=%f, PSF=%v}",
		s.Center.X, s.Center.Y, s.StarBoundingBox, s.Background, s.MeanBrightness, s.PeakBrightness, s.Flux, s.HFR, s.PSF)
}

// AddOffset returns a copy of the star with the given offset applied.
func (s *Star) AddOffset(xOffset, yOffset int) *Star {
	return &Star{
		Center:         Point2d{X: s.Center.X + float64(xOffset), Y: s.Center.Y + float64(yOffset)},
		StarBoundingBox: s.StarBoundingBox.Add(image.Pt(xOffset, yOffset)),
		Background:     s.Background,
		MeanBrightness: s.MeanBrightness,
		PeakBrightness: s.PeakBrightness,
		Flux:           s.Flux,
		HFR:            s.HFR,
		PSF:            s.PSF,
	}
}

// PSFModel contains the result of PSF fitting.
type PSFModel struct {
	PSFType      StarDetectorPSFFitType
	OffsetX      float64
	OffsetY      float64
	Peak         float64
	Background   float64
	SigmaX       float64
	SigmaY       float64
	Sigma        float64
	FWHMx        float64
	FWHMy        float64
	ThetaRadians float64
	FWHMPixels   float64
	FWHMArcsecs  float64
	Eccentricity float64
	RSquared     float64
}

// NewPSFModel creates a PSFModel with computed derived fields.
func NewPSFModel(
	psfType StarDetectorPSFFitType,
	offsetX, offsetY, peak, background,
	sigmaX, sigmaY, fwhmX, fwhmY, thetaRadians,
	rSquared, pixelScale float64,
) *PSFModel {
	a := math.Max(fwhmX, fwhmY)
	b := math.Min(fwhmX, fwhmY)
	eccentricity := math.Sqrt(1 - b*b/(a*a))
	fwhmPixels := math.Sqrt(fwhmX * fwhmY)

	return &PSFModel{
		PSFType:      psfType,
		OffsetX:      offsetX,
		OffsetY:      offsetY,
		Peak:         peak,
		Background:   background,
		SigmaX:       sigmaX,
		SigmaY:       sigmaY,
		Sigma:        math.Sqrt(sigmaX * sigmaY),
		FWHMx:        fwhmX,
		FWHMy:        fwhmY,
		ThetaRadians: thetaRadians,
		Eccentricity: eccentricity,
		FWHMPixels:   fwhmPixels,
		FWHMArcsecs:  fwhmPixels * pixelScale,
		RSquared:     rSquared,
	}
}

func (p *PSFModel) String() string {
	return fmt.Sprintf("{PSFType=%s, OffsetX=%f, OffsetY=%f, Peak=%f, Background=%f, SigmaX=%f, SigmaY=%f, FWHMx=%f, FWHMy=%f, FWHMPixels=%f, FWHMArcsecs=%f, Eccentricity=%f, RSquared=%f}",
		p.PSFType, p.OffsetX, p.OffsetY, p.Peak, p.Background, p.SigmaX, p.SigmaY, p.FWHMx, p.FWHMy, p.FWHMPixels, p.FWHMArcsecs, p.Eccentricity, p.RSquared)
}

// StarDetectorMetrics tracks detection filtering statistics.
type StarDetectorMetrics struct {
	StructureCandidates int
	TotalDetected       int
	TooSmall            int
	OnBorder            int
	TooDistortedBounds  []image.Rectangle
	DegenerateBounds    []image.Rectangle
	SaturatedBounds     []image.Rectangle
	LowSensitivityBounds []image.Rectangle
	NotCenteredBounds   []image.Rectangle
	TooFlatBounds       []image.Rectangle
	TooLowHFR           int
	HFRAnalysisFailed   int
	PSFFitFailed        int
	OutsideROI          int
	SaturatedPixelCount int64
	HotpixelCount       int64
}

// NewStarDetectorMetrics creates an initialized StarDetectorMetrics.
func NewStarDetectorMetrics() *StarDetectorMetrics {
	return &StarDetectorMetrics{
		TooDistortedBounds:  make([]image.Rectangle, 0),
		DegenerateBounds:    make([]image.Rectangle, 0),
		SaturatedBounds:     make([]image.Rectangle, 0),
		LowSensitivityBounds: make([]image.Rectangle, 0),
		NotCenteredBounds:   make([]image.Rectangle, 0),
		TooFlatBounds:       make([]image.Rectangle, 0),
	}
}

func (m *StarDetectorMetrics) AddROIOffset(xOffset, yOffset int) {
	offset := image.Pt(xOffset, yOffset)
	allBounds := []*[]image.Rectangle{
		&m.TooDistortedBounds,
		&m.DegenerateBounds,
		&m.SaturatedBounds,
		&m.LowSensitivityBounds,
		&m.NotCenteredBounds,
		&m.TooFlatBounds,
	}
	for _, bounds := range allBounds {
		for i, r := range *bounds {
			(*bounds)[i] = r.Add(offset)
		}
	}
}

// DebugData contains optional debug information from detection.
type DebugData struct {
	// 0 = No structure map at that pixel, 1 = Original, 2 = Dilated
	StructureMap []byte
	DetectionROI image.Rectangle
}

// StarDetectorResult is the output of the detection pipeline.
type StarDetectorResult struct {
	DetectedStars []*Star
	Metrics       *StarDetectorMetrics
	DebugData     *DebugData
}

// ZonePosition identifies a zone in the 3x3 field grid.
type ZonePosition int

const (
	ZoneTopLeft ZonePosition = iota
	ZoneTop
	ZoneTopRight
	ZoneLeft
	ZoneCenter
	ZoneRight
	ZoneBottomLeft
	ZoneBottom
	ZoneBottomRight
)

// ZoneData holds per-zone statistics.
type ZoneData struct {
	Label      string
	MedianHFR  float64
	MedianFWHM float64
	StarCount  int
}

// FieldAnalysis holds the result of 3x3 field tilt analysis.
type FieldAnalysis struct {
	Zones       map[ZonePosition]ZoneData
	TiltPct     float64
	OffAxisPct  float64
	BestCorner  string
	WorstCorner string
	Reliable    bool
}
