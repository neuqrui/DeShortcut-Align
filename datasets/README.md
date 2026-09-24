# Datasets

Raw JSON shipped with this release. Parquet files under `processed/` are built by the training scripts.

| File | Role |
|------|------|
| `risk_data_7B.json` | Harmful prompts for the 7B and Qwen3-4B RL pools |
| `risk_data_14B.json` | Harmful prompts for the 14B RL pool |
| `agca_benign_rl.json` | Intention-inverted benign prompts mixed into RL (AGCA) |
| `STAR-1-harmful-sft.json` | Harmful SFT demonstrations |
| `agca_benign_sft.json` | Benign SFT demonstrations paired with the harmful set |

Rebuild the benign files with `scripts/run_tss_agca.py` and `scripts/agca/generate_responses.py` if you want a fresh AGCA sample.
