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

// CvImageStatisticsFlags controls which statistics to compute.
type CvImageStatisticsFlags int

const (
	StatNone   CvImageStatisticsFlags = 0
	StatMedian CvImageStatisticsFlags = 1
	StatMAD    CvImageStatisticsFlags = 2
	StatMean   CvImageStatisticsFlags = 4
	StatStdDev CvImageStatisticsFlags = 8
	StatAll    CvImageStatisticsFlags = StatMedian | StatMAD | StatMean | StatStdDev
)

// CvImageStatistics holds image statistics results.
type CvImageStatistics struct {
	Median float64
	MAD    float64
	Mean   float64
	StdDev float64
}

func (s CvImageStatistics) String() string {
	return fmt.Sprintf("{Median=%f, MAD=%f, Mean=%f, StdDev=%f}", s.Median, s.MAD, s.Mean, s.StdDev)
}

// Ranged represents a value range.
type Ranged struct {
	Start float64
	End   float64
}

// KappaSigmaResult holds noise estimation results.
type KappaSigmaResult struct {
	Sigma          float64
	BackgroundMean float64
	NumIterations  int
}

// ToFloat32Mat converts a uint16 pixel array to a CV_32F Mat normalized to [0, 1].
func ToFloat32Mat(pixels []uint16, bpp, width, height int) Mat {
	data := NewMatWithSize(height, width)
	dest := data.DataFloat32()
	scalingRatio := float32(uint32(1) << uint(bpp))
	numPixels := width * height
	for i := 0; i < numPixels; i++ {
		dest[i] = float32(pixels[i]) / scalingRatio
	}
	return data
}

// ConvolveGaussian applies a separated Gaussian convolution.
func ConvolveGaussian(src, dst *Mat, kernelSize int) {
	if kernelSize < 3 || kernelSize%2 == 0 {
		panic("kernelSize must be a positive odd number >= 3")
	}
	sigma := 0.159758 * float64(kernelSize)
	kernel := getGaussianKernel1D(kernelSize, sigma)
	defer kernel.Close()
	sepFilter2DReflect(*src, dst, kernel, kernel)
}

// CalculateStatisticsHistogram computes statistics using a histogram approach.
func CalculateStatisticsHistogram(
	img Mat,
	useLogHistogram bool,
	valueRange *Ranged,
	rect *image.Rectangle,
	flags CvImageStatisticsFlags,
) CvImageStatistics {
	var result CvImageStatistics
	numBuckets := 1 << 16
	histogram := make([]uint32, numBuckets)
	bucketLowerBounds := make([]float64, numBuckets)
	firstLogValue := math.Log(1.0 / float64(numBuckets) / 2.0)
	logBucketSize := (0.0 - firstLogValue) / float64(numBuckets)

	if useLogHistogram {
		bucketLowerBounds[0] = 0
		nextLogValue := firstLogValue
		for i := 1; i < numBuckets; i++ {
			bucketLowerBounds[i] = math.Exp(nextLogValue)
			nextLogValue += logBucketSize
		}
	} else {
		for i := 0; i < numBuckets; i++ {
			bucketLowerBounds[i] = float64(i) / float64(numBuckets)
		}
	}

	var numPixels int64
	imgWidth := img.Cols()
	height, width := img.Rows(), img.Cols()
	startX, startY := 0, 0
	if rect != nil {
		startX, startY = rect.Min.X, rect.Min.Y
		width, height = rect.Dx(), rect.Dy()
	}

	data := img.DataFloat32()

	if valueRange != nil {
		for row := 0; row < height; row++ {
			rowOffset := (startY+row)*imgWidth + startX
			for col := 0; col < width; col++ {
				pixelValue := float64(data[rowOffset+col])
				if pixelValue < valueRange.Start || pixelValue >= valueRange.End {
					continue
				}
				bucketIndex := computeBucketIndex(pixelValue, useLogHistogram, firstLogValue, logBucketSize, numBuckets)
				histogram[bucketIndex]++
				numPixels++
			}
		}
	} else {
		for row := 0; row < height; row++ {
			rowOffset := (startY+row)*imgWidth + startX
			for col := 0; col < width; col++ {
				pixelValue := float64(data[rowOffset+col])
				bucketIndex := computeBucketIndex(pixelValue, useLogHistogram, firstLogValue, logBucketSize, numBuckets)
				histogram[bucketIndex]++
			}
		}
		numPixels = int64(height) * int64(width)
	}

	if flags&StatMedian != 0 || flags&StatMAD != 0 {
		targetMedianCount := float64(numPixels) / 2.0
		var currentCount uint32
		medianPosition := -1
		for i := 0; i < numBuckets; i++ {
			currentCount += histogram[i]
			if float64(currentCount) >= targetMedianCount {
				interpolationRatio := (float64(currentCount) - targetMedianCount) / float64(histogram[i])
				nextBucket := 1.0
				if i < numBuckets-1 {
					nextBucket = bucketLowerBounds[i+1]
				}
				result.Median = bucketLowerBounds[i] + (nextBucket-bucketLowerBounds[i])*interpolationRatio
				medianPosition = i
				break
			}
		}

		if flags&StatMAD != 0 {
			upIndex := medianPosition
			downIndex := medianPosition - 1
			currentCount = 0
			for {
				upDist := math.MaxFloat64
				if upIndex < numBuckets {
					upDist = math.Abs(bucketLowerBounds[upIndex] - result.Median)
				}
				downDist := math.MaxFloat64
				if downIndex >= 0 {
					downDist = math.Abs(bucketLowerBounds[downIndex] - result.Median)
				}
				var chosenIndex int
				if upDist <= downDist {
					chosenIndex = upIndex
					upIndex++
				} else {
					chosenIndex = downIndex
					downIndex--
				}
				currentCount += histogram[chosenIndex]
				if float64(currentCount) >= targetMedianCount {
					result.MAD = math.Abs(bucketLowerBounds[chosenIndex] - result.Median)
					break
				}
			}
		}
	}

	if flags&StatMean != 0 || flags&StatStdDev != 0 {
		var pixelTotal float64
		for i := 0; i < numBuckets; i++ {
			pixelTotal += float64(histogram[i]) * bucketLowerBounds[i]
		}
		result.Mean = pixelTotal / float64(numPixels)

		if flags&StatStdDev != 0 {
			var sse float64
			for i := 0; i < numBuckets; i++ {
				err := bucketLowerBounds[i] - result.Mean
				sse += float64(histogram[i]) * (err * err)
			}
			result.StdDev = math.Sqrt(sse / float64(numPixels-1))
		}
	}
	return result
}

func computeBucketIndex(pixelValue float64, useLog bool, firstLogValue, logBucketSize float64, numBuckets int) int {
	var idx int
	if useLog {
		idx = int(math.Ceil((math.Log(pixelValue) - firstLogValue) / logBucketSize))
	} else {
		idx = int(math.Floor(pixelValue * float64(numBuckets)))
	}
	if idx < 0 {
		idx = 0
	}
	if idx >= numBuckets {
		idx = numBuckets - 1
	}
	return idx
}

// GetB3SplineFilter creates a B3 spline wavelet filter for the given dyadic layer.
func GetB3SplineFilter(dyadicLayer int) Mat {
	size := (1 << uint(dyadicLayer+2)) + 1
	filter := NewMatWithSize(size, 1)
	data := filter.DataFloat32()
	data[0] = 0.0625
	data[size-1] = 0.0625
	data[1<<uint(dyadicLayer)] = 0.25
	data[size-(1<<uint(dyadicLayer))-1] = 0.25
	data[size>>1] = 0.375
	return filter
}

// ClampInPlace clamps mat values to [min, max].
func ClampInPlace(src *Mat, min, max float32) {
	data := src.DataFloat32()
	n := src.Rows() * src.Cols()
	for i := 0; i < n; i++ {
		if data[i] < min {
			data[i] = min
		} else if data[i] > max {
			data[i] = max
		}
	}
}

// SubtractInPlace subtracts rhs from lhs in place, clamping to [0, 1].
func SubtractInPlace(lhs, rhs *Mat) {
	lhsData := lhs.DataFloat32()
	rhsData := rhs.DataFloat32()
	n := lhs.Rows() * lhs.Cols()
	for i := 0; i < n; i++ {
		lhsData[i] -= rhsData[i]
	}
	ClampInPlace(lhs, 0.0, 1.0)
}

// ComputeResidualAtrousB3SplineDyadicWaveletLayer computes wavelet residual.
func ComputeResidualAtrousB3SplineDyadicWaveletLayer(src Mat, numLayers int) Mat {
	previousLayer := src
	tempMat := NewMat()
	for i := 0; i < numLayers; i++ {
		scalingFilter := GetB3SplineFilter(i)
		sepFilter2DReflect(previousLayer, &tempMat, scalingFilter, scalingFilter)
		scalingFilter.Close()
		previousLayer = tempMat
	}
	return previousLayer
}

// BilinearSamplePixelValue samples a pixel value using bilinear interpolation.
func BilinearSamplePixelValue(img Mat, y, x float64) float64 {
	y0 := int(math.Floor(y))
	y1 := y0 + 1
	if y1 > img.Rows()-1 {
		y1 = img.Rows() - 1
	}
	x0 := int(math.Floor(x))
	x1 := x0 + 1
	if x1 > img.Cols()-1 {
		x1 = img.Cols() - 1
	}
	yRatio := y - float64(y0)
	xRatio := x - float64(x0)

	data := img.DataFloat32()
	width := img.Cols()
	p00 := float64(data[y0*width+x0])
	p01 := float64(data[y0*width+x1])
	p10 := float64(data[y1*width+x0])
	p11 := float64(data[y1*width+x1])
	interpolatedX0 := p00 + xRatio*(p01-p00)
	interpolatedX1 := p10 + xRatio*(p11-p10)
	return interpolatedX0 + yRatio*(interpolatedX1-interpolatedX0)
}

// KappaSigmaNoiseEstimate performs iterative kappa-sigma noise estimation.
func KappaSigmaNoiseEstimate(img Mat, clippingMultiplier float64, allowedError float64, maxIterations int) KappaSigmaResult {
	maskMat := NewMat()
	defer maskMat.Close()

	threshold := float32(math.MaxFloat32)
	lastSigma := 1.0
	lastBackgroundMean := 1.0
	numIterations := 0

	for numIterations < maxIterations {
		var meanVal, sigmaVal float64

		if numIterations > 0 {
			inRangeScalar(img, math.SmallestNonzeroFloat32, threshold-math.SmallestNonzeroFloat32, &maskMat)
			meanVal, sigmaVal = meanStdDevWithMask(img, maskMat)
		} else {
			meanVal, sigmaVal = matMeanStdDev(img)
		}

		numIterations++
		if numIterations > 1 {
			if math.Abs(sigmaVal-lastSigma) <= allowedError {
				lastSigma = sigmaVal
				break
			}
		}
		threshold = float32(meanVal + clippingMultiplier*sigmaVal)
		lastSigma = sigmaVal
		lastBackgroundMean = meanVal
	}

	return KappaSigmaResult{
		Sigma:          lastSigma,
		BackgroundMean: lastBackgroundMean,
		NumIterations:  numIterations,
	}
}

// meanStdDevWithMask computes mean and stddev of pixels where mask is non-zero.
func meanStdDevWithMask(img Mat, mask Mat) (float64, float64) {
	imgData := img.DataFloat32()
	maskData := mask.DataFloat32()
	numPixels := img.Rows() * img.Cols()

	var sum float64
	var count int64
	for i := 0; i < numPixels; i++ {
		if maskData[i] != 0 {
			sum += float64(imgData[i])
			count++
		}
	}
	if count == 0 {
		return 0, 0
	}
	mean := sum / float64(count)

	var sse float64
	for i := 0; i < numPixels; i++ {
		if maskData[i] != 0 {
			diff := float64(imgData[i]) - mean
			sse += diff * diff
		}
	}
	return mean, math.Sqrt(sse / float64(count))
}

// Binarize thresholds the image.
func Binarize(src, dst *Mat, threshold float64) {
	thresholdBinary(*src, dst, float32(threshold), 1.0)
}
