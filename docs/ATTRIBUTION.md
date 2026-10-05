# Attribution

All code in this repository is original and MIT-licensed. It builds on the open-source
libraries, models and data listed here. Datasets, with their licences and measured row
counts, are listed separately in [`data/SOURCES.md`](../data/SOURCES.md).

## Libraries (runtime)

| Library | Licence | Use |
| --- | --- | --- |
| FastAPI, Starlette | MIT, BSD-3 | REST + WebSocket API |
| Pydantic, pydantic-settings | MIT | Typed schema and config |
| Uvicorn | BSD-3 | ASGI server |
| HTTPX | BSD-3 | Async HTTP for connectors |
| feedparser | BSD-2 | RSS/Atom parsing |
| PyYAML | MIT | Source configuration |
| prometheus-client | Apache-2.0 | Metrics |
| pandas, PyArrow | BSD-3, Apache-2.0 | Data handling, Parquet |
| DuckDB | MIT | Demo signal store |
| orjson | Apache-2.0 / MIT | Fast JSON |

## Libraries (development)

pytest, pytest-asyncio (MIT, Apache-2.0), ruff (MIT), mypy (MIT).

## Data services

- **FRED** (Federal Reserve Bank of St. Louis): rates, Moody's Baa yields, oil and dollar index (keyless CSV).
- **SEC EDGAR full-text search**: 8-K item codes for credit labels.

- **The GDELT Project** (gdeltproject.org): GKG 2.1 and Events 2.0, free and open.
- **SEC EDGAR**: public filings, accessed per the SEC fair-access policy.
- **Hugging Face Hub**: dataset hosting, keyless downloads.

## Models

| Model | Licence | Use |
| --- | --- | --- |
| [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) (Araci, 2019) | Apache-2.0 | Base for the English sentiment model |
| [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) (Conneau et al., 2020) | MIT | Base for the multilingual sentiment and event models |
| [MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) (Laurer et al., 2023) | MIT | Zero-shot event fallback |
| [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (NLLB Team, 2022) | CC BY-NC 4.0 | Offline translate-train data generation only (not shipped) |
| [fastText lid.176](https://fasttext.cc/docs/en/language-identification.html) (Joulin et al., 2016) | CC BY-SA 3.0 | Language identification |

## Libraries (NLP)

| Library | Licence | Use |
| --- | --- | --- |
| ONNX Runtime | MIT | CPU inference, INT8 quantisation |
| Hugging Face tokenizers | Apache-2.0 | Runtime tokenisation |
| PyTorch, Transformers, Accelerate (training only) | BSD-3, Apache-2.0 | Fine-tuning, ONNX export |
| datasketch | MIT | MinHash LSH de-duplication |
| pyahocorasick | BSD-3 | Multilingual alias matching |
| fasttext-numpy2-wheel | MIT | fastText bindings compatible with NumPy 2 |
| scikit-learn, SciPy, joblib | BSD-3 | Impact model (HistGradientBoosting), metrics, statistics |
| yfinance (training only) | Apache-2.0 | Keyless daily prices for impact labels |
| pysentiment2 (eval only) | MIT | Loughran-McDonald dictionary baseline |
| vaderSentiment (eval only) | MIT | VADER lexicon baseline |

## Lexicons

- **Loughran-McDonald Master Dictionary** (Loughran & McDonald, 2011, *Journal of Finance*), via `pysentiment2`: evaluation baseline only.
- **VADER** (Hutto & Gilbert, 2014): evaluation baseline only.

## Reference data

- **Wikidata** (CC0): entity labels, aliases and demonyms in 20+ languages (`data/reference/gazetteer.json.gz`).
