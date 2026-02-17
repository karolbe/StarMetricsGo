//go:build purego || js

package starmetrics

import (
	"image"
	"math"
	"sort"
)

// Mat is a pure Go 2D float32 matrix.
type Mat struct {
	data    []float32
	rows    int
	cols    int
	stride  int // elements per row in backing array (may differ from cols for sub-matrices)
	dataOff int // offset into data for sub-matrices
	owned   bool
}

func NewMat() Mat { return Mat{} }

func NewMatWithSize(rows, cols int) Mat {
	return Mat{
		data:   make([]float32, rows*cols),
		rows:   rows,
		cols:   cols,
		stride: cols,
		owned:  true,
	}
}

func (m Mat) Rows() int   { return m.rows }
func (m Mat) Cols() int   { return m.cols }
func (m Mat) Empty() bool { return m.data == nil || m.rows == 0 || m.cols == 0 }

func (m Mat) Clone() Mat {
	newData := make([]float32, m.rows*m.cols)
	for r := 0; r < m.rows; r++ {
		srcOff := m.dataOff + r*m.stride
		copy(newData[r*m.cols:], m.data[srcOff:srcOff+m.cols])
	}
	return Mat{data: newData, rows: m.rows, cols: m.cols, stride: m.cols, owned: true}
}

func (m *Mat) Close() {
	if m.owned {
		m.data = nil
	}
	m.rows = 0
	m.cols = 0
}

// DataFloat32 returns the backing float32 slice.
// Only valid for contiguous mats (not un-cloned sub-matrices from Region).
func (m Mat) DataFloat32() []float32 {
	return m.data[m.dataOff:]
}

func (m Mat) Region(r image.Rectangle) Mat {
	return Mat{
		data:    m.data,
		rows:    r.Dy(),
		cols:    r.Dx(),
		stride:  m.stride,
		dataOff: m.dataOff + r.Min.Y*m.stride + r.Min.X,
		owned:   false,
	}
}

func (m *Mat) SetToZero() {
	for r := 0; r < m.rows; r++ {
		off := m.dataOff + r*m.stride
		for c := 0; c < m.cols; c++ {
			m.data[off+c] = 0
		}
	}
}

func CopyMatTo(src Mat, dst *Mat) {
	if dst.rows != src.rows || dst.cols != src.cols || dst.data == nil {
		*dst = NewMatWithSize(src.rows, src.cols)
	}
	for r := 0; r < src.rows; r++ {
		srcOff := src.dataOff + r*src.stride
		dstOff := dst.dataOff + r*dst.stride
		copy(dst.data[dstOff:dstOff+src.cols], src.data[srcOff:srcOff+src.cols])
	}
}

// --- Pure Go CV operations ---

func reflectIndex(idx, size int) int {
	if idx < 0 {
		idx = -idx
	}
	for idx >= size {
		idx = 2*size - 2 - idx
		if idx < 0 {
			idx = -idx
		}
	}
	return idx
}

func sepFilter2DReflect(src Mat, dst *Mat, kernelX, kernelY Mat) {
	rows, cols := src.rows, src.cols
	srcData := src.DataFloat32()
	kx := kernelX.DataFloat32()
	ky := kernelY.DataFloat32()
	kxLen := kernelX.rows * kernelX.cols
	kyLen := kernelY.rows * kernelY.cols
	kxHalf := kxLen / 2
	kyHalf := kyLen / 2

	if dst.rows != rows || dst.cols != cols || dst.data == nil {
		*dst = NewMatWithSize(rows, cols)
	}

	temp := make([]float32, rows*cols)

	// Horizontal pass — split into border and interior
	for r := 0; r < rows; r++ {
		rowOff := r * cols
		// Left border
		for c := 0; c < kxHalf && c < cols; c++ {
			var sum float32
			for k := 0; k < kxLen; k++ {
				cc := reflectIndex(c+k-kxHalf, cols)
				sum += srcData[rowOff+cc] * kx[k]
			}
			temp[rowOff+c] = sum
		}
		// Interior — no bounds check needed
		for c := kxHalf; c < cols-kxHalf; c++ {
			var sum float32
			base := rowOff + c - kxHalf
			for k := 0; k < kxLen; k++ {
				sum += srcData[base+k] * kx[k]
			}
			temp[rowOff+c] = sum
		}
		// Right border
		for c := cols - kxHalf; c < cols; c++ {
			if c < kxHalf {
				continue // already handled in left border for tiny images
			}
			var sum float32
			for k := 0; k < kxLen; k++ {
				cc := reflectIndex(c+k-kxHalf, cols)
				sum += srcData[rowOff+cc] * kx[k]
			}
			temp[rowOff+c] = sum
		}
	}

	// Vertical pass — pre-compute row offsets to avoid multiply in inner loop
	dstData := dst.DataFloat32()
	rowOffs := make([]int, kyLen)

	// Top border rows
	for r := 0; r < kyHalf && r < rows; r++ {
		for k := 0; k < kyLen; k++ {
			rowOffs[k] = reflectIndex(r+k-kyHalf, rows) * cols
		}
		dstOff := r * cols
		for c := 0; c < cols; c++ {
			var sum float32
			for k := 0; k < kyLen; k++ {
				sum += temp[rowOffs[k]+c] * ky[k]
			}
			dstData[dstOff+c] = sum
		}
	}
	// Interior rows
	for r := kyHalf; r < rows-kyHalf; r++ {
		for k := 0; k < kyLen; k++ {
			rowOffs[k] = (r + k - kyHalf) * cols
		}
		dstOff := r * cols
		for c := 0; c < cols; c++ {
			var sum float32
			for k := 0; k < kyLen; k++ {
				sum += temp[rowOffs[k]+c] * ky[k]
			}
			dstData[dstOff+c] = sum
		}
	}
	// Bottom border rows
	for r := rows - kyHalf; r < rows; r++ {
		if r < kyHalf {
			continue
		}
		for k := 0; k < kyLen; k++ {
			rowOffs[k] = reflectIndex(r+k-kyHalf, rows) * cols
		}
		dstOff := r * cols
		for c := 0; c < cols; c++ {
			var sum float32
			for k := 0; k < kyLen; k++ {
				sum += temp[rowOffs[k]+c] * ky[k]
			}
			dstData[dstOff+c] = sum
		}
	}
}

func getGaussianKernel1D(size int, sigma float64) Mat {
	m := NewMatWithSize(size, 1)
	data := m.DataFloat32()
	half := size / 2
	sum := 0.0
	for i := 0; i < size; i++ {
		x := float64(i - half)
		val := math.Exp(-x * x / (2 * sigma * sigma))
		data[i] = float32(val)
		sum += val
	}
	for i := range data[:size] {
		data[i] = float32(float64(data[i]) / sum)
	}
	return m
}

func medianBlur(src Mat, dst *Mat, ksize int) {
	rows, cols := src.rows, src.cols
	srcData := src.DataFloat32()
	result := make([]float32, rows*cols)

	if ksize == 3 {
		// Fast path: sorting network for 9 elements
		for r := 0; r < rows; r++ {
			r0 := r - 1
			r1 := r
			r2 := r + 1
			if r0 < 0 {
				r0 = 0
			}
			if r2 >= rows {
				r2 = rows - 1
			}
			row0 := r0 * cols
			row1 := r1 * cols
			row2 := r2 * cols
			for c := 0; c < cols; c++ {
				c0 := c - 1
				c2 := c + 1
				if c0 < 0 {
					c0 = 0
				}
				if c2 >= cols {
					c2 = cols - 1
				}
				// Load 3x3 neighborhood
				a := srcData[row0+c0]
				b := srcData[row0+c]
				cc := srcData[row0+c2]
				d := srcData[row1+c0]
				e := srcData[row1+c]
				f := srcData[row1+c2]
				g := srcData[row2+c0]
				h := srcData[row2+c]
				ii := srcData[row2+c2]
				// Sorting network for median of 9 (Bose-Nelson)
				if a > b {
					a, b = b, a
				}
				if d > e {
					d, e = e, d
				}
				if g > h {
					g, h = h, g
				}
				if a > d {
					a, d = d, a
				}
				if b > e {
					b, e = e, b
				}
				if d > g {
					d, g = g, d
				}
				if e > h {
					e, h = h, e
				}
				if cc > f {
					cc, f = f, cc
				}
				if f > ii {
					f, ii = ii, f
				}
				if cc > f {
					cc, f = f, cc
				}
				if a > cc {
					a, cc = cc, a
				}
				if b > f {
					b, f = f, b
				}
				if d > cc {
					d, cc = cc, d
				}
				if e > f {
					e, f = f, e
				}
				if d > b {
					d, b = b, d
				}
				if g > cc {
					g, cc = cc, g
				}
				if e > cc {
					e, cc = cc, e
				}
				if e > d {
					e, d = d, e
				}
				_ = a
				_ = b
				_ = cc
				_ = f
				_ = g
				_ = h
				_ = ii
				result[r*cols+c] = e
			}
		}
	} else {
		half := ksize / 2
		neighbors := make([]float32, ksize*ksize)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				idx := 0
				for dr := -half; dr <= half; dr++ {
					for dc := -half; dc <= half; dc++ {
						rr, cc := r+dr, c+dc
						if rr < 0 {
							rr = 0
						}
						if rr >= rows {
							rr = rows - 1
						}
						if cc < 0 {
							cc = 0
						}
						if cc >= cols {
							cc = cols - 1
						}
						neighbors[idx] = srcData[rr*cols+cc]
						idx++
					}
				}
				sort.Slice(neighbors[:idx], func(i, j int) bool { return neighbors[i] < neighbors[j] })
				result[r*cols+c] = neighbors[idx/2]
			}
		}
	}

	if dst.rows != rows || dst.cols != cols || dst.data == nil {
		*dst = NewMatWithSize(rows, cols)
	}
	copy(dst.DataFloat32(), result)
}

func absDiff(a, b Mat, dst *Mat) {
	n := a.rows * a.cols
	ad, bd := a.DataFloat32(), b.DataFloat32()
	if dst.rows != a.rows || dst.cols != a.cols || dst.data == nil {
		*dst = NewMatWithSize(a.rows, a.cols)
	}
	dd := dst.DataFloat32()
	for i := 0; i < n; i++ {
		d := ad[i] - bd[i]
		if d < 0 {
			d = -d
		}
		dd[i] = d
	}
}

func thresholdBinary(src Mat, dst *Mat, thresh, maxval float32) {
	n := src.rows * src.cols
	sd := src.DataFloat32()
	if dst.rows != src.rows || dst.cols != src.cols || dst.data == nil {
		*dst = NewMatWithSize(src.rows, src.cols)
	}
	dd := dst.DataFloat32()
	for i := 0; i < n; i++ {
		if sd[i] > thresh {
			dd[i] = maxval
		} else {
			dd[i] = 0
		}
	}
}

func countNonZero(src Mat) int {
	data := src.DataFloat32()
	n := src.rows * src.cols
	count := 0
	for i := 0; i < n; i++ {
		if data[i] != 0 {
			count++
		}
	}
	return count
}

func morphDilateEllipse(src Mat, dst *Mat, kernelSize, iterations int) {
	rows, cols := src.rows, src.cols
	half := kernelSize / 2

	type off struct{ dr, dc int }
	var offsets []off
	for dr := -half; dr <= half; dr++ {
		for dc := -half; dc <= half; dc++ {
			nr := float64(dr) / float64(half)
			nc := float64(dc) / float64(half)
			if nr*nr+nc*nc <= 1.0 {
				offsets = append(offsets, off{dr, dc})
			}
		}
	}

	if dst.rows != rows || dst.cols != cols || dst.data == nil {
		*dst = NewMatWithSize(rows, cols)
	}

	current := make([]float32, rows*cols)
	copy(current, src.DataFloat32())
	result := make([]float32, rows*cols)

	for iter := 0; iter < iterations; iter++ {
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				maxVal := current[r*cols+c]
				for _, o := range offsets {
					rr := reflectIndex(r+o.dr, rows)
					cc := reflectIndex(c+o.dc, cols)
					if v := current[rr*cols+cc]; v > maxVal {
						maxVal = v
					}
				}
				result[r*cols+c] = maxVal
			}
		}
		current, result = result, current
	}
	copy(dst.DataFloat32(), current)
}

func inRangeScalar(src Mat, lower, upper float32, dst *Mat) {
	n := src.rows * src.cols
	sd := src.DataFloat32()
	if dst.rows != src.rows || dst.cols != src.cols || dst.data == nil {
		*dst = NewMatWithSize(src.rows, src.cols)
	}
	dd := dst.DataFloat32()
	for i := 0; i < n; i++ {
		if sd[i] >= lower && sd[i] <= upper {
			dd[i] = 1.0
		} else {
			dd[i] = 0
		}
	}
}

func matMeanStdDev(src Mat) (float64, float64) {
	data := src.DataFloat32()
	n := src.rows * src.cols
	if n == 0 {
		return 0, 0
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(data[i])
	}
	mean := sum / float64(n)
	var sse float64
	for i := 0; i < n; i++ {
		d := float64(data[i]) - mean
		sse += d * d
	}
	return mean, math.Sqrt(sse / float64(n))
}

func matCopyToWithMask(src Mat, dst *Mat, mask Mat) {
	n := src.rows * src.cols
	sd, dd, md := src.DataFloat32(), dst.DataFloat32(), mask.DataFloat32()
	for i := 0; i < n; i++ {
		if md[i] != 0 {
			dd[i] = sd[i]
		}
	}
}

func imWriteMat(_ string, _ Mat) {
	// No-op in pure Go build (debug image saving not supported)
}

func imReadMat(_ string) Mat {
	return Mat{} // Not supported; use FITS reader
}

func matConvertToFloat(src Mat, dst *Mat) {
	CopyMatTo(src, dst)
}
