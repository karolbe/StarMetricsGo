.PHONY: all native purego wasm serve clean

all: native purego wasm

native:
	go build -o bin/starmetrics ./cmd/starmetrics

purego:
	go build -tags purego -o bin/starmetrics-pure ./cmd/starmetrics

wasm:
	GOOS=js GOARCH=wasm go build -o web/hfr.wasm ./cmd/wasm
	cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" web/wasm_exec.js

serve: wasm
	cd web && python3 -m http.server 8080

clean:
	rm -f bin/starmetrics bin/starmetrics-pure web/hfr.wasm
