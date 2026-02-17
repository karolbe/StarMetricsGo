/*
Extracted from HocusFocus plugin by George Hilios.
Original Copyright © 2021 George Hilios <ghilios+NINA@googlemail.com>
Licensed under Mozilla Public License 2.0.
Ported to Go.
*/

package starmetrics

import "math"

var sigmaToFWHM = 2.0 * math.Sqrt(2.0*math.Log(2.0))

// FitStar fits a 2D Gaussian PSF model to a star.
func FitStar(star *Star, srcImage Mat, psfResolution int, pixelScale, goodnessThreshold float64) *PSFModel {
	bb := star.StarBoundingBox
	background := star.Background
	nominalBBWidth := math.Sqrt(float64(bb.Dx()) * float64(bb.Dy()))
	samplingSize := nominalBBWidth / float64(psfResolution)

	bbLeft := float64(bb.Min.X)
	bbTop := float64(bb.Min.Y)
	bbRight := float64(bb.Max.X)
	bbBottom := float64(bb.Max.Y)
	bbWidth := float64(bb.Dx())
	bbHeight := float64(bb.Dy())

	startX := star.Center.X - samplingSize*math.Floor((star.Center.X-bbLeft)/samplingSize)
	startY := star.Center.Y - samplingSize*math.Floor((star.Center.Y-bbTop)/samplingSize)
	endX := bbRight
	endY := bbBottom
	widthPixels := int(math.Floor((endX-startX-1e-4)/samplingSize)) + 1
	heightPixels := int(math.Floor((endY-startY-1e-4)/samplingSize)) + 1
	numPixels := widthPixels * heightPixels
	if numPixels < 7 {
		return nil
	}

	centroidBrightness := BilinearSamplePixelValue(srcImage, star.Center.Y, star.Center.X)

	inputs := make([][]float64, 0, numPixels)
	outputs := make([]float64, 0, numPixels)
	for py := startY; py < endY; py += samplingSize {
		for px := startX; px < endX; px += samplingSize {
			value := BilinearSamplePixelValue(srcImage, py, px)
			inputs = append(inputs, []float64{px - star.Center.X, py - star.Center.Y})
			outputs = append(outputs, value)
		}
	}
	numPixels = len(inputs)
	if numPixels < 7 {
		return nil
	}

	sigmaUpperBound := math.Sqrt(bbWidth*bbWidth+bbHeight*bbHeight) / 2.0
	centroidAboveBg := math.Max(0.0, centroidBrightness-background)
	dxLimit := bbWidth / 8.0
	dyLimit := bbHeight / 8.0

	x0 := []float64{centroidAboveBg, background, 0.0, 0.0, bbWidth / 3.0, bbHeight / 3.0, 0.0}
	lower := []float64{0.0, 0.0, -dxLimit, -dyLimit, 0, 0, -math.Pi / 2.0}
	upper := []float64{2.0, 1.0, dxLimit, dyLimit, sigmaUpperBound, sigmaUpperBound, math.Pi / 2.0}
	scale := []float64{0.01, 0.01, 0.1, 0.1, 1, 1, 1}

	solution := levenbergMarquardt(inputs, outputs, x0, lower, upper, scale, 1e-8, 200)
	if solution == nil {
		return nil
	}

	sigX := solution[4]
	sigY := solution[5]
	if math.IsNaN(sigX) || math.IsNaN(sigY) {
		return nil
	}

	theta := euclidianModulus(solution[6], math.Pi)
	if theta > math.Pi/2.0 {
		theta -= math.Pi
	}
	theta = -theta

	if sigY > sigX {
		if theta < 0 {
			theta += math.Pi / 2.0
		} else {
			theta -= math.Pi / 2.0
		}
		sigX, sigY = sigY, sigX
	}

	rSquared := computeRSquared(inputs, outputs, solution)
	if rSquared < goodnessThreshold {
		return nil
	}

	fwhmX := sigX * sigmaToFWHM
	fwhmY := sigY * sigmaToFWHM

	return NewPSFModel(
		PSFFitGaussian,
		solution[2], solution[3],
		solution[0], solution[1],
		sigX, sigY,
		fwhmX, fwhmY,
		theta,
		rSquared,
		pixelScale,
	)
}

func euclidianModulus(x, y float64) float64 {
	return math.Mod(math.Mod(x, y)+y, y)
}

func gaussianValue(p, input []float64) float64 {
	A, B := p[0], p[1]
	x, y := input[0], input[1]
	x0, y0 := p[2], p[3]
	U, V, T := p[4], p[5], p[6]

	cosT, sinT := math.Cos(T), math.Sin(T)
	X := (x-x0)*cosT + (y-y0)*sinT
	Y := -(x-x0)*sinT + (y-y0)*cosT
	E := X*X/(2*U*U) + Y*Y/(2*V*V)
	return B + A*math.Exp(-E)
}

func gaussianGradient(p, input, grad []float64) {
	A := p[0]
	x, y := input[0], input[1]
	x0, y0 := p[2], p[3]
	U, V, T := p[4], p[5], p[6]

	cosT, sinT := math.Cos(T), math.Sin(T)
	X := (x-x0)*cosT + (y-y0)*sinT
	Y := -(x-x0)*sinT + (y-y0)*cosT
	X2 := X * X
	Y2 := Y * Y
	U2 := U * U
	U3 := U2 * U
	V2 := V * V
	V3 := V2 * V
	E := X2/(2*U2) + Y2/(2*V2)
	eE := math.Exp(-E)

	grad[0] = eE
	grad[1] = 1.0
	grad[2] = A * (cosT*X/U2 - sinT*Y/V2) * eE
	grad[3] = A * (sinT*X/U2 + cosT*Y/V2) * eE
	grad[4] = A * X2 / U3 * eE
	grad[5] = A * Y2 / V3 * eE
	grad[6] = A * X * Y * (1.0/V2 - 1.0/U2) * eE
}

func computeRSquared(inputs [][]float64, outputs, p []float64) float64 {
	yBar := 0.0
	for _, o := range outputs {
		yBar += o
	}
	yBar /= float64(len(outputs))

	tss, rss := 0.0, 0.0
	for i := range inputs {
		est := gaussianValue(p, inputs[i])
		res := est - outputs[i]
		disp := outputs[i] - yBar
		rss += res * res
		tss += disp * disp
	}
	if tss > 0 {
		return 1.0 - rss/tss
	}
	return 0.0
}

func levenbergMarquardt(
	inputs [][]float64, outputs,
	x0, lower, upper, scale []float64,
	tolerance float64, maxIter int,
) []float64 {
	n := len(x0)
	m := len(inputs)

	x := make([]float64, n)
	copy(x, x0)
	for j := 0; j < n; j++ {
		x[j] = clampLM(x[j], lower[j], upper[j])
	}

	fi := make([]float64, m)
	jac := make([][]float64, m)
	for i := range jac {
		jac[i] = make([]float64, n)
	}
	grad := make([]float64, n)

	computeResidualsAndJacobian(inputs, outputs, x, fi, jac, grad, m, n)
	cost := sumOfSquares(fi)

	lambda := 1e-3
	nu := 2.0

	JtJ := make([][]float64, n)
	for i := range JtJ {
		JtJ[i] = make([]float64, n)
	}
	Jtf := make([]float64, n)
	A := make([][]float64, n)
	for i := range A {
		A[i] = make([]float64, n)
	}
	rhs := make([]float64, n)
	dx := make([]float64, n)
	xNew := make([]float64, n)
	fiNew := make([]float64, m)

	for iter := 0; iter < maxIter; iter++ {
		for i := 0; i < n; i++ {
			Jtf[i] = 0
			for j := 0; j < n; j++ {
				JtJ[i][j] = 0
			}
		}
		for k := 0; k < m; k++ {
			for i := 0; i < n; i++ {
				ji := jac[k][i]
				Jtf[i] += ji * fi[k]
				for j := i; j < n; j++ {
					JtJ[i][j] += ji * jac[k][j]
				}
			}
		}
		for i := 0; i < n; i++ {
			for j := 0; j < i; j++ {
				JtJ[i][j] = JtJ[j][i]
			}
		}

		gradNorm := 0.0
		for i := 0; i < n; i++ {
			gradNorm += Jtf[i] * Jtf[i]
		}
		if math.Sqrt(gradNorm) < tolerance*cost {
			break
		}

		for tries := 0; tries < 20; tries++ {
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					A[i][j] = JtJ[i][j]
				}
				A[i][i] += lambda * (scale[i] * scale[i])
				rhs[i] = -Jtf[i]
			}

			if !solveLinear(A, rhs, dx, n) {
				lambda *= nu
				continue
			}

			for j := 0; j < n; j++ {
				xNew[j] = clampLM(x[j]+dx[j], lower[j], upper[j])
			}

			for k := 0; k < m; k++ {
				fiNew[k] = gaussianValue(xNew, inputs[k]) - outputs[k]
			}
			costNew := sumOfSquares(fiNew)

			if costNew < cost {
				improvement := (cost - costNew) / cost
				copy(x, xNew)
				copy(fi, fiNew)
				cost = costNew
				lambda = math.Max(lambda/3.0, 1e-15)
				nu = 2.0

				computeResidualsAndJacobian(inputs, outputs, x, fi, jac, grad, m, n)

				if improvement < tolerance {
					return x
				}
				break
			} else {
				lambda *= nu
				nu *= 2.0
				if lambda > 1e16 {
					return x
				}
			}
		}
	}
	return x
}

func computeResidualsAndJacobian(
	inputs [][]float64, outputs,
	x, fi []float64, jac [][]float64, grad []float64,
	m, n int,
) {
	for k := 0; k < m; k++ {
		fi[k] = gaussianValue(x, inputs[k]) - outputs[k]
		gaussianGradient(x, inputs[k], grad)
		for j := 0; j < n; j++ {
			jac[k][j] = grad[j]
		}
	}
}

func sumOfSquares(fi []float64) float64 {
	s := 0.0
	for _, v := range fi {
		s += v * v
	}
	return s
}

func clampLM(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func solveLinear(A [][]float64, b, x []float64, n int) bool {
	a := make([][]float64, n)
	for i := range a {
		a[i] = make([]float64, n)
		copy(a[i], A[i])
	}
	rhsCopy := make([]float64, n)
	copy(rhsCopy, b)

	for col := 0; col < n; col++ {
		maxRow := col
		maxVal := math.Abs(a[col][col])
		for row := col + 1; row < n; row++ {
			av := math.Abs(a[row][col])
			if av > maxVal {
				maxVal = av
				maxRow = row
			}
		}
		if maxVal < 1e-30 {
			return false
		}

		if maxRow != col {
			a[col], a[maxRow] = a[maxRow], a[col]
			rhsCopy[col], rhsCopy[maxRow] = rhsCopy[maxRow], rhsCopy[col]
		}

		pivot := a[col][col]
		for row := col + 1; row < n; row++ {
			factor := a[row][col] / pivot
			for j := col; j < n; j++ {
				a[row][j] -= factor * a[col][j]
			}
			rhsCopy[row] -= factor * rhsCopy[col]
		}
	}

	for row := n - 1; row >= 0; row-- {
		sum := rhsCopy[row]
		for j := row + 1; j < n; j++ {
			sum -= a[row][j] * x[j]
		}
		x[row] = sum / a[row][row]
	}
	return true
}
