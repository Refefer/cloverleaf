# Building cloverleaf's Python bindings, and the GPU trainer's dependencies.
#
#   make python    build and install the bindings into a virtualenv, optimised for this CPU
#   make gpu       the same, plus torch and the GPU trainer's other dependencies
#   make wheel     a portable release wheel in dist/
#   make test      the Rust, Python and GPU test suites
#
# The virtualenv is the active one ($VIRTUAL_ENV) if there is one, else .venv here, created on
# first use; override with VENV=path.  Needs Rust 1.85+ and Python 3.9+.

VENV ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV),.venv)
PYTHON ?= python3
BIN := $(VENV)/bin

.PHONY: help python gpu wheel test test-rust test-python test-gpu clean

help:
	@sed -n '3,9p' Makefile | sed 's/^# \{0,1\}//'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

$(BIN)/maturin: | $(BIN)/python
	$(BIN)/pip install --upgrade pip "maturin>=1.9.4" numpy pytest

python: | $(BIN)/maturin
	VIRTUAL_ENV=$(abspath $(VENV)) RUSTFLAGS="-C target-cpu=native" $(BIN)/maturin develop --release
	@echo "Built cloverleaf into $(VENV); activate it with: source $(VENV)/bin/activate"

gpu: python
	$(BIN)/pip install -r gpu/requirements.txt

wheel: | $(BIN)/maturin
	$(BIN)/maturin build --release --out dist

test: test-rust test-python test-gpu

test-rust:
	cargo test --release

test-python: python
	$(BIN)/python -m pytest -q tests/python

test-gpu: gpu
	cd gpu && $(abspath $(BIN))/python -m pytest -q tests

clean:
	cargo clean
	rm -rf dist
