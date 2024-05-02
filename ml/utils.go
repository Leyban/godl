package ml

// Matrix Dot Product
func Dot(A, B [][]float64) [][]float64 {

	P := make([][]float64, len(A))

	for i := range P {
		P[i] = make([]float64, len(B[0]))
	}

	for i := range A {
		for j := range B[0] {
			for k := range B {
				P[i][j] += A[i][k] * B[k][j]
			}
		}
	}

	return P
}

func Transpose(Mat [][]float64) [][]float64 {
	n, m := len(Mat), len(Mat[0])
	t := make([][]float64, m)
	for i := range t {
		t[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			t[j][i] = Mat[i][j]
		}
	}
	return t
}

// Sum a matrix along a specified axis.
func Sum(m [][]float64, axis int) []float64 {
	var sum []float64
	switch axis {
	case 0:
		for i := range m {
			var rowSum float64
			for _, v := range m[i] {
				rowSum += v
			}
			sum = append(sum, rowSum)
		}
	case 1:
		for i := 0; i < len(m[0]); i++ {
			var colSum float64
			for _, row := range m {
				colSum += row[i]
			}
			sum = append(sum, colSum)
		}
	}
	return sum
}

// Scale an array by a factor of c
func Scale(s []float64, c float64) []float64 {
	for i := range s {
		s[i] *= c
	}
	return s
}
