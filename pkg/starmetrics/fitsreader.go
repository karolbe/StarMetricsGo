package starmetrics

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

// FitsMetadata holds parsed FITS header key-value pairs.
type FitsMetadata struct {
	Headers map[string]string
}

// NewFitsMetadata creates an empty FitsMetadata.
func NewFitsMetadata() *FitsMetadata {
	return &FitsMetadata{Headers: make(map[string]string)}
}

func (m *FitsMetadata) GetString(key string) string {
	if v, ok := m.Headers[strings.ToUpper(key)]; ok {
		return v
	}
	return ""
}

func (m *FitsMetadata) GetDouble(key string) (float64, bool) {
	v, ok := m.Headers[strings.ToUpper(key)]
	if !ok {
		return 0, false
	}
	d, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil {
		return 0, false
	}
	return d, true
}

func (m *FitsMetadata) GetInt(key string) (int, bool) {
	v, ok := m.Headers[strings.ToUpper(key)]
	if !ok {
		return 0, false
	}
	i, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil {
		return 0, false
	}
	return i, true
}

func (m *FitsMetadata) GetDateTime(key string) (time.Time, bool) {
	v, ok := m.Headers[strings.ToUpper(key)]
	if !ok {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(v))
	if err != nil {
		return time.Time{}, false
	}
	return t, true
}

// Convenience accessors matching C# FitsMetadata properties.
func (m *FitsMetadata) ObjectName() string  { return m.GetString("OBJECT") }
func (m *FitsMetadata) ImageType() string   { return m.GetString("IMAGETYP") }
func (m *FitsMetadata) CameraName() string  { return m.GetString("INSTRUME") }
func (m *FitsMetadata) Filter() string      { return m.GetString("FILTER") }
func (m *FitsMetadata) TelescopeName() string { return m.GetString("TELESCOP") }

func (m *FitsMetadata) ExposureTime() (float64, bool) {
	if v, ok := m.GetDouble("EXPTIME"); ok {
		return v, true
	}
	return m.GetDouble("EXPOSURE")
}

func (m *FitsMetadata) FocalLength() (float64, bool)  { return m.GetDouble("FOCALLEN") }
func (m *FitsMetadata) PixelSizeX() (float64, bool)   { return m.GetDouble("XPIXSZ") }
func (m *FitsMetadata) PixelSizeY() (float64, bool)   { return m.GetDouble("YPIXSZ") }

// FitsImageData holds parsed FITS image data.
type FitsImageData struct {
	Pixels   []uint16
	Width    int
	Height   int
	BitDepth int
	Metadata *FitsMetadata
}

// ReadFits reads FITS headers and pixel data from a file.
func ReadFits(filePath string) (*FitsImageData, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("opening FITS file: %w", err)
	}
	defer f.Close()
	return readFitsFromReader(f, false)
}

// ReadFitsMetadataOnly reads only FITS headers without loading pixel data.
func ReadFitsMetadataOnly(filePath string) (*FitsImageData, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("opening FITS file: %w", err)
	}
	defer f.Close()
	return readFitsFromReader(f, true)
}

// ReadFitsFromBytes reads FITS headers and pixel data from a byte slice.
func ReadFitsFromBytes(data []byte) (*FitsImageData, error) {
	return readFitsFromReader(bytes.NewReader(data), false)
}

func readFitsFromReader(r io.Reader, skipPixelData bool) (*FitsImageData, error) {

	var bitpix, naxis, width, height int
	bzero := 0.0
	bscale := 1.0
	headerDone := false
	metadata := NewFitsMetadata()

	recordBuf := make([]byte, 80)

	for !headerDone {
		for i := 0; i < 36; i++ {
			_, err := io.ReadFull(r, recordBuf)
			if err != nil {
				return nil, fmt.Errorf("reading FITS header record: %w", err)
			}
			record := string(recordBuf)
			keyword := strings.TrimSpace(record[:8])

			if keyword == "END" {
				headerDone = true
				remaining := 35 - i
				if remaining > 0 {
					skipBuf := make([]byte, remaining*80)
					io.ReadFull(r, skipBuf)
				}
				break
			}

			if len(record) > 10 && record[8] == '=' && record[9] == ' ' {
				rawValue := strings.TrimSpace(strings.SplitN(record[10:], "/", 2)[0])
				parsedValue := parseFitsValue(rawValue)

				if keyword != "" && parsedValue != "" {
					metadata.Headers[strings.ToUpper(keyword)] = parsedValue
				}

				switch keyword {
				case "BITPIX":
					bitpix, _ = strconv.Atoi(strings.TrimSpace(rawValue))
				case "NAXIS":
					naxis, _ = strconv.Atoi(strings.TrimSpace(rawValue))
				case "NAXIS1":
					width, _ = strconv.Atoi(strings.TrimSpace(rawValue))
				case "NAXIS2":
					height, _ = strconv.Atoi(strings.TrimSpace(rawValue))
				case "BZERO":
					bzero, _ = strconv.ParseFloat(strings.TrimSpace(rawValue), 64)
				case "BSCALE":
					bscale, _ = strconv.ParseFloat(strings.TrimSpace(rawValue), 64)
				}
			}
		}
	}

	if naxis < 2 || width == 0 || height == 0 {
		return nil, fmt.Errorf("invalid FITS: NAXIS=%d, NAXIS1=%d, NAXIS2=%d", naxis, width, height)
	}

	effectiveBpp := 16
	if bitpix == 8 {
		effectiveBpp = 8
	}

	if skipPixelData {
		return &FitsImageData{
			Pixels:   nil,
			Width:    width,
			Height:   height,
			BitDepth: effectiveBpp,
			Metadata: metadata,
		}, nil
	}

	numPixels := width * height
	pixels := make([]uint16, numPixels)

	switch bitpix {
	case 16:
		rawBytes := make([]byte, numPixels*2)
		if _, err := io.ReadFull(r, rawBytes); err != nil {
			return nil, fmt.Errorf("reading 16-bit pixel data: %w", err)
		}
		for i := 0; i < numPixels; i++ {
			signedVal := int16(binary.BigEndian.Uint16(rawBytes[i*2:]))
			physicalVal := float64(signedVal)*bscale + bzero
			pixels[i] = uint16(clampFloat64(physicalVal, 0, 65535))
		}

	case -32:
		rawBytes := make([]byte, numPixels*4)
		if _, err := io.ReadFull(r, rawBytes); err != nil {
			return nil, fmt.Errorf("reading -32 float pixel data: %w", err)
		}
		for i := 0; i < numPixels; i++ {
			intBits := binary.BigEndian.Uint32(rawBytes[i*4:])
			floatVal := math.Float32frombits(intBits)
			physicalVal := float64(floatVal)*bscale + bzero
			pixels[i] = uint16(clampFloat64(physicalVal, 0, 65535))
		}

	case 8:
		rawBytes := make([]byte, numPixels)
		if _, err := io.ReadFull(r, rawBytes); err != nil {
			return nil, fmt.Errorf("reading 8-bit pixel data: %w", err)
		}
		for i := 0; i < numPixels; i++ {
			physicalVal := float64(rawBytes[i])*bscale + bzero
			pixels[i] = uint16(clampFloat64(physicalVal, 0, 65535))
		}

	case 32:
		rawBytes := make([]byte, numPixels*4)
		if _, err := io.ReadFull(r, rawBytes); err != nil {
			return nil, fmt.Errorf("reading 32-bit pixel data: %w", err)
		}
		for i := 0; i < numPixels; i++ {
			intVal := int32(binary.BigEndian.Uint32(rawBytes[i*4:]))
			physicalVal := float64(intVal)*bscale + bzero
			pixels[i] = uint16(clampFloat64(physicalVal, 0, 65535))
		}

	default:
		return nil, fmt.Errorf("unsupported BITPIX: %d", bitpix)
	}

	return &FitsImageData{
		Pixels:   pixels,
		Width:    width,
		Height:   height,
		BitDepth: effectiveBpp,
		Metadata: metadata,
	}, nil
}

func clampFloat64(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func parseFitsValue(rawValue string) string {
	if rawValue == "" {
		return ""
	}
	if rawValue == "T" {
		return "True"
	}
	if rawValue == "F" {
		return "False"
	}
	if strings.HasPrefix(rawValue, "'") {
		endQuote := strings.LastIndex(rawValue, "'")
		if endQuote > 0 {
			return strings.TrimRight(rawValue[1:endQuote], " ")
		}
		return strings.TrimLeft(strings.TrimRight(rawValue, " "), "'")
	}
	return rawValue
}
