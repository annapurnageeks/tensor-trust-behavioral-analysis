# Tensor Trust Behavioral Analysis

Exploratory analysis of player strategy persistence and strategy exposure in the public Tensor Trust prompt-injection game dataset.

**Author:** Apil Adhikari

**ORCID:** [0009-0008-4475-0838](https://orcid.org/0009-0008-4475-0838)

## Research questions

1. Do players repeat a detected strategy more often after a successful attack than after a failed attack?
2. Is earlier exposure to a strategy associated with a player's first use of that strategy?

The analysis is observational. It does not call an LLM, generate attack prompts, or establish causal effects.

## Main results

- After cleaning, 551,005 attack rows remain from the supplied 563,349-row v2 dump.
- Among 540,971 within-session consecutive pairs, strategy repetition was 56.84% after success and 52.46% after failure. The difference was 4.38 percentage points (95% player-bootstrap CI: 1.45–7.32); the independent-label-set chance baseline was 27.96%.
- The repeat-rate difference was similar with 30-, 60-, and 120-minute session gaps. A calendar-week fixed-effect model with player-clustered errors estimated an odds ratio of 1.229 (95% CI: 1.104–1.368).
- Adoption comparisons depend strongly on exposure timing. The main table classifies an adopter exposed only after first use as not previously exposed; it should not be interpreted causally. In a stricter pre-first-attack cohort, only 28 players were exposed before their first outgoing attack, so those estimates are imprecise.
- Eight rule-based labels are multi-label, and 35.11% of attacks match none of the detectors. “Other” is heterogeneous, not a single strategy.

See [`results.md`](results.md) for methods, detailed findings, limitations, and links to all aggregate results.

## Dataset

This analysis uses the official [Tensor Trust data repository](https://github.com/HumanCompatibleAI/tensor-trust-data), specifically the raw-data v2 files:

- [Attack dump](https://github.com/HumanCompatibleAI/tensor-trust-data/blob/main/raw-data/v2/raw_dump_attacks.jsonl.bz2)
- [Defense dump](https://github.com/HumanCompatibleAI/tensor-trust-data/blob/main/raw-data/v2/raw_dump_defenses.jsonl.bz2) (provided for reference; this script reads the attack dump only)

Download `raw_dump_attacks.jsonl.bz2` into `data/raw_dump_attacks.jsonl.bz2`. The script expects that exact path. **Do not commit the raw dumps:** they contain attack/defense text, access codes, and player IDs. This repository excludes the `data/` directory.

The supplied v2 attack file has 563,349 rows, while the 2023 paper reports 126,808 attacks and 46,457 defenses for its release. The v2 label, later dates, and unique local IDs support a later-expanded-dump explanation, but the repository does not publish a count reconciliation; the difference is not fully verified.

## Run locally

Tested with Python 3.14.7. From the repository directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
# Put raw_dump_attacks.jsonl.bz2 at data/raw_dump_attacks.jsonl.bz2
python analysis/main_analysis.py
```

The script uses a fixed seed (`2023`). It writes aggregate CSV tables under `outputs/` and PNG charts under `figures/`. Main execution does not print or save attack text. Raw inputs, the virtual environment, LaTeX build intermediates, and the draft paper source/PDF are excluded from this repository.

## Included outputs

- `cleaning_summary.csv`: row counts after cleaning steps.
- `table1_coverage.csv`: label coverage counts and percentages.
- `table2_persistence.csv`: conditional repetition rates and player-bootstrap intervals.
- `table3_sensitivity.csv`: 30-, 60-, and 120-minute session-gap checks.
- `table4_adoption.csv`: exposure-group adoption rates, intervals, and tests.
- `stage3_calendar_adjusted.csv`: calendar-week-adjusted repetition model.
- `stage4_pre_exposure_sensitivity.csv`: later use among players exposed before their first attack.
- `weekly_strategy_popularity.csv`: weekly strategy shares.
- `figures/figure1_repetition.png`, `figures/figure2_adoption.png`, and `figures/figure3_weekly.png`: aggregate charts corresponding to the analyses.

## Citation

For the dataset and original game study, cite:

> Toyer, S., Watkins, O., Mendes, E. A., Svegliato, J., Bailey, L., Wang, T., Ong, I., Elmaaroufi, K., Abbeel, P., Darrell, T., Ritter, A., & Russell, S. (2023). [*Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game*](https://arxiv.org/abs/2311.01011). arXiv:2311.01011. https://doi.org/10.48550/arXiv.2311.01011

To cite this software/repository, use the metadata in [`CITATION.cff`](CITATION.cff).
