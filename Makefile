# Sentinel - one-command workflows. Python 3.11, Node 20.
# `make setup && make demo` is the no-Docker path the README documents.

PY      ?= python3.11
VENV    ?= .venv
BIN     := $(VENV)/bin
PORT    ?= 8000
SPEED   ?= 30
# Put src/ on the path explicitly: macOS can flag the editable-install .pth file hidden,
# and Python skips hidden .pth files.
export PYTHONPATH := $(CURDIR)/src

.DEFAULT_GOAL := help
.PHONY: help setup setup-py models models-all web demo live record download train export calibrate impact gazetteer package-models test test-all lint typecheck keyscan eval bench docker clean deck diagrams

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

setup-py: $(BIN)/python  ## Install pinned Python deps into .venv
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt
	$(BIN)/pip install -e . --no-deps

models:  ## Download fine-tuned models (checksummed, resumable) + language ID (all four models)
	@mkdir -p models
	@[ -f models/lid.176.ftz ] || curl -fL -o models/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
	$(BIN)/python -m engine.models_download

models-all:  ## Same as models (kept for compatibility)
	$(BIN)/python -m engine.models_download --all

setup: setup-py models  ## Install Python deps, download models, build the web UI (if Node is present)
	@if command -v npm >/dev/null 2>&1 && [ -f src/web/package.json ]; then $(MAKE) web; \
	 else echo "npm not found or web app absent: API-only mode (docs at /docs)"; fi

web:  ## Build the React dashboard into src/web/dist
	cd src/web && npm ci && npm run build

demo:  ## Offline demo: replay every recorded source at $(SPEED)x on http://localhost:$(PORT)
	SENTINEL_MODE=replay SENTINEL_REPLAY_SPEED=$(SPEED) $(BIN)/uvicorn engine.api.app:app --host 0.0.0.0 --port $(PORT)

live:  ## Live keyless sources, each falling back to its replay twin on failure
	SENTINEL_MODE=live $(BIN)/uvicorn engine.api.app:app --host 0.0.0.0 --port $(PORT)

record:  ## Re-record replay twins from live sources (needs internet; ~10 min)
	$(BIN)/python -m engine.ingest.record --minutes 10
	$(BIN)/python -m engine.ingest.record --gdelt-backfill-hours 6 --max-docs-per-file 250

download:  ## Download public datasets (training sets, tweet replay) and regenerate data/SOURCES.md
	$(BIN)/python -m training.download_data all

gazetteer:  ## Rebuild the multilingual entity gazetteer from Wikidata
	$(BIN)/python -m training.build_gazetteer

train:  ## Full training run: data, translate-train, weak labels, 3 fine-tunes, ONNX export (GPU/MPS recommended)
	$(BIN)/pip install -r requirements-train.txt
	$(BIN)/python -m training.download_data training
	$(BIN)/python -m training.weak_labels --windows 700
	$(BIN)/python -m training.translate phrasebank
	$(BIN)/python -m training.translate events
	$(BIN)/python -m training.train_sentiment en
	$(BIN)/python -m training.train_sentiment multi
	$(BIN)/python -m training.train_events
	$(MAKE) export

export:  ## Export fine-tuned models to ONNX (fidelity-checked quantisation) + calibrate sentiment
	$(BIN)/python -m training.export_onnx sentiment-en sentiment-multi event-multi --zero-shot

calibrate:  ## Re-fit the sentiment score maps (FiQA + range check)
	$(BIN)/python -m training.calibrate_sentiment

impact:  ## Impact model: prices + 2,000 GDELT windows (2019-2026) -> features/labels -> train + evaluate
	$(BIN)/python -m training.impact_data all
	$(BIN)/python -m training.train_impact

credit:  ## Credit Event work: distant supervision, back-translation, retrain, tune on validation, test once
	$(BIN)/python -m training.credit_labels all
	$(BIN)/python -m training.translate credit
	$(BIN)/python -m training.train_events --out event-multi
	$(BIN)/python -m training.export_onnx event-multi
	$(BIN)/python -m training.credit_tune tune

modules: stress-book backtest  ## Rebuild Module A and Module B data and results

backtest:  ## Module A: backtest 2020-2026 with costs, benchmarks and 100 placebos -> docs/eval/backtest.json
	$(BIN)/python -m modules.rebalancer.backtest

stress-book:  ## Module B: sector mix, shock library (yfinance/FRED) and the seeded synthetic book
	$(BIN)/python -m modules.stress.calibrate
	$(BIN)/python -m modules.stress.portfolio

replay-2022:  ## Module B: 24 Feb 2022 through the full system -> docs/eval/replay_2022-02-24.json
	$(BIN)/python -m modules.stress.replay

BASE_URL ?= https://github.com/yugh88/VIT_YughJuneja_Hackathon/releases/download/models-v1
package-models:  ## Pack models/onnx into release archives + write the checksum manifest
	$(BIN)/python scripts/package_models.py --base-url $(BASE_URL)

test:  ## Run the test suite (offline)
	$(BIN)/pytest

test-all:  ## Tests including the ones that hit the internet
	$(BIN)/pytest -m "network or slow or not (network or slow)"

lint:  ## ruff + mypy + API-key scan
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests
	$(BIN)/mypy
	$(BIN)/python scripts/check_no_keys.py

typecheck:
	$(BIN)/mypy

keyscan:  ## Fail if anything in the repo looks like an API key
	$(BIN)/python scripts/check_no_keys.py

eval:  ## Measure model quality and write docs/model_card.md
	$(BIN)/python -m training.evaluate

bench:  ## Measure latency, throughput and Monte Carlo runtime
	$(BIN)/python -m engine.bench

docker:  ## Build and run the demo container
	docker compose up --build

deck:  ## Rebuild docs/presentation.pdf + .pptx from docs/deck/deck.html (needs Chrome, python-pptx)
	PYTHON="$(CURDIR)/$(BIN)/python" docs/deck/build.sh

diagrams:  ## Re-render the hand-drawn diagrams in docs/diagrams (needs Node 20, Chrome)
	docs/diagrams/source/build.sh

clean-scratch:  ## Empty scratch/ (all experiment artefacts live there)
	find scratch -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
	@df -h . | tail -1

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache src/web/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
