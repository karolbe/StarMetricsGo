//go:build !purego && !js

package starmetrics

import (
	"image"

	"gocv.io/x/gocv"
)

// Mat wraps gocv.Mat for the native OpenCV backend.
type Mat struct {
	m gocv.Mat
}

func NewMat() Mat                        { return Mat{m: gocv.NewMat()} }
func NewMatWithSize(rows, cols int) Mat   { return Mat{m: gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F)} }
func (mat Mat) Rows() int               { return mat.m.Rows() }
func (mat Mat) Cols() int               { return mat.m.Cols() }
func (mat Mat) Empty() bool             { return mat.m.Empty() }
func (mat Mat) Clone() Mat              { return Mat{m: mat.m.Clone()} }
func (mat *Mat) Close()                 { mat.m.Close() }
func (mat Mat) Region(r image.Rectangle) Mat { return Mat{m: mat.m.Region(r)} }

func (mat Mat) DataFloat32() []float32 {
	data, _ := mat.m.DataPtrFloat32()
	return data
}

func (mat *Mat) SetToZero() {
	mat.m.SetTo(gocv.NewScalar(0, 0, 0, 0))
}

func CopyMatTo(src Mat, dst *Mat) {
	src.m.CopyTo(&dst.m)
}

// --- CV operations ---

func sepFilter2DReflect(src Mat, dst *Mat, kernelX, kernelY Mat) {
	gocv.SepFilter2D(src.m, &dst.m, gocv.MatTypeCV32F, kernelX.m, kernelY.m, image.Pt(-1, -1), 0, gocv.BorderReflect)
}

func getGaussianKernel1D(size int, sigma float64) Mat {
	return Mat{m: gocv.GetGaussianKernel(size, sigma)}
}

func medianBlur(src Mat, dst *Mat, ksize int) {
	gocv.MedianBlur(src.m, &dst.m, ksize)
}

func absDiff(a, b Mat, dst *Mat) {
	gocv.AbsDiff(a.m, b.m, &dst.m)
}

func thresholdBinary(src Mat, dst *Mat, thresh, maxval float32) {
	gocv.Threshold(src.m, &dst.m, thresh, maxval, gocv.ThresholdBinary)
}

func countNonZero(src Mat) int {
	return gocv.CountNonZero(src.m)
}

func morphDilateEllipse(src Mat, dst *Mat, kernelSize, iterations int) {
	kernel := gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(kernelSize, kernelSize))
	defer kernel.Close()
	gocv.MorphologyExWithParams(src.m, &dst.m, gocv.MorphDilate, kernel, iterations, gocv.BorderReflect)
}

func inRangeScalar(src Mat, lower, upper float32, dst *Mat) {
	lo := gocv.NewMatFromScalar(gocv.NewScalar(float64(lower), 0, 0, 0), gocv.MatTypeCV32F)
	defer lo.Close()
	hi := gocv.NewMatFromScalar(gocv.NewScalar(float64(upper), 0, 0, 0), gocv.MatTypeCV32F)
	defer hi.Close()
	mask8 := gocv.NewMat()
	defer mask8.Close()
	gocv.InRange(src.m, lo, hi, &mask8)
	// InRange outputs CV_8U; convert to CV_32F so DataFloat32() works
	mask8.ConvertTo(&dst.m, gocv.MatTypeCV32F)
}

func matMeanStdDev(src Mat) (float64, float64) {
	meanMat := gocv.NewMat()
	defer meanMat.Close()
	stdMat := gocv.NewMat()
	defer stdMat.Close()
	gocv.MeanStdDev(src.m, &meanMat, &stdMat)
	return meanMat.GetDoubleAt(0, 0), stdMat.GetDoubleAt(0, 0)
}

func matCopyToWithMask(src Mat, dst *Mat, mask Mat) {
	mask8 := gocv.NewMat()
	defer mask8.Close()
	mask.m.ConvertTo(&mask8, gocv.MatTypeCV8U)
	src.m.CopyToWithMask(&dst.m, mask8)
}

func imWriteMat(path string, m Mat) {
	gocv.IMWrite(path, m.m)
}

func imReadMat(path string) Mat {
	return Mat{m: gocv.IMRead(path, gocv.IMReadUnchanged)}
}

func matConvertToFloat(src Mat, dst *Mat) {
	src.m.ConvertTo(&dst.m, gocv.MatTypeCV32F)
}
