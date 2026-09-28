"""Stage-by-stage analysis of the Tensor Trust attack data."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, fisher_exact
import statsmodels.api as sm
import statsmodels.formula.api as smf


# A fixed seed makes future random samples and bootstrap results repeatable.
RANDOM_SEED = 2023
np.random.seed(RANDOM_SEED)

PROJECT_DIR = Path(__file__).resolve().parent
ATTACKS_FILE = PROJECT_DIR / "data" / "raw_dump_attacks.jsonl.bz2"
OUTPUT_DIR = PROJECT_DIR / "outputs"
SESSION_GAP_MINUTES = 60
BOOTSTRAP_REPLICATES = 2000

# These intentionally broad rules are transparent starting points, not a
# substitute for manually reviewing a sample of the labeled attacks.
STRATEGY_PATTERNS = {
    "repeated_character_prefix": re.compile(r"^(.)\1{2,}", re.DOTALL),
    "artisanlib": re.compile(r"\bartisanlib\b", re.IGNORECASE),
    "role_play_framing": re.compile(
        r"\b(?:pretend|roleplay|role-play|act as|imagine you are|"
        r"you are now|in this (?:fictional )?(?:story|scenario|game))\b",
        re.IGNORECASE,
    ),
    "instruction_override": re.compile(
        r"\b(?:ignore|disregard|forget|override|replace|update)\b"
        r".{0,60}\b(?:previous|prior|above|earlier)\b"
        r".{0,30}\b(?:instructions?|rules?|directions?|prompt)\b|"
        r"\b(?:previous|prior|above|earlier)\b"
        r".{0,30}\b(?:instructions?|rules?|directions?|prompt)\b"
        r".{0,60}\b(?:ignore|disregard|forget|override|replace|update)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "direct_request": re.compile(
        r"\b(?:please\s+)?(?:tell|show|give|provide|reveal|say|write|"
        r"print|output|respond|return|share)\b|"
        r"\b(?:what|who|where|when|how)\s+(?:is|are|was|were|do|does|can)\b",
        re.IGNORECASE,
    ),
    "code_or_pseudocode": re.compile(
        r"\b(?:code|pseudocode|source code|program|script|function|algorithm)\b",
        re.IGNORECASE,
    ),
    "encoded_input": re.compile(
        r"\b(?:base64|binary|hex(?:adecimal)?|rot-?13|morse|cipher|"
        r"decode|encoded|encoding)\b|[01]{8,}|[A-Za-z0-9+/]{24,}={0,2}",
        re.IGNORECASE,
    ),
    "few_shot_examples": re.compile(
        r"\bfew[- ]shot\b|\bexamples?\b|"
        r"\b(?:Q|Question)\s*:.{0,200}\b(?:A|Answer)\s*:|"
        r"\b(?:user|assistant)\s*:",
        re.IGNORECASE | re.DOTALL,
    ),
}


def clean_attacks(raw_attacks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a copy of the attacks and return it with an aggregate count report."""
    original_row_count = len(raw_attacks)
    print(f"Rows loaded: {original_row_count:,}")

    # Work on a copy so neither the source file nor the original table changes.
    attacks = raw_attacks.copy()
    report_rows = [{"step": "loaded", "rows_remaining": len(attacks)}]

    # Convert timestamps explicitly so sorting uses dates, not text strings.
    attacks["timestamp"] = pd.to_datetime(
        attacks["timestamp"], utc=True, errors="coerce"
    )
    invalid_timestamps = int(attacks["timestamp"].isna().sum())
    print(
        f"Rows after timestamp conversion: {len(attacks):,}; "
        f"invalid timestamps: {invalid_timestamps:,}"
    )
    if invalid_timestamps:
        raise ValueError("Some timestamps could not be converted; inspect before continuing.")
    report_rows.append({"step": "timestamps_converted", "rows_remaining": len(attacks)})

    # Stable sorting preserves input order when timestamps are equal.
    attacks = attacks.sort_values("timestamp", kind="stable")
    if not attacks["timestamp"].is_monotonic_increasing:
        raise AssertionError("Timestamp sorting did not produce chronological order.")
    print(f"Rows after sorting by time: {len(attacks):,}")
    report_rows.append({"step": "sorted_by_time", "rows_remaining": len(attacks)})

    # Remove a row only if all its columns exactly match another row.
    duplicate_count = int(attacks.duplicated().sum())
    attacks = attacks.drop_duplicates().copy()
    print(
        f"Exact duplicate rows removed: {duplicate_count:,}; "
        f"rows remaining: {len(attacks):,}"
    )
    if attacks.duplicated().any():
        raise AssertionError("Exact duplicate rows remain after duplicate removal.")
    report_rows.append({"step": "exact_duplicates_removed", "rows_remaining": len(attacks)})

    # Use the dataset's existing Boolean success label when it is available.
    if "output_is_access_granted" in attacks.columns:
        attacks["success"] = attacks["output_is_access_granted"].astype(bool)
        success_source = "output_is_access_granted"
    else:
        raise KeyError("No existing success column is available in this dataset.")
    print(
        f"Rows with success labels: {len(attacks):,}; "
        f"successes: {int(attacks['success'].sum()):,}; source: {success_source}"
    )
    report_rows.append({"step": "success_labeled", "rows_remaining": len(attacks)})

    # Compare the attack with its row's access code literally, ignoring case.
    attack_text = attacks["attacker_input"].fillna("").astype(str).str.casefold()
    access_codes = attacks["access_code"].fillna("").astype(str).str.casefold()
    attacks["contains_own_access_code"] = pd.Series(
        (bool(code) and code in text for text, code in zip(attack_text, access_codes)),
        index=attacks.index,
        dtype=bool,
    )

    flagged_count = int(attacks["contains_own_access_code"].sum())
    attacks = attacks.loc[~attacks["contains_own_access_code"]].copy()
    print(f"Rows containing the defender's own access code: {flagged_count:,}")
    print(f"Rows remaining after exclusion: {len(attacks):,}")
    report_rows.append({"step": "own_access_code_excluded", "rows_remaining": len(attacks)})

    # Check that cleaning changed only the working copy, with expected row loss.
    if len(raw_attacks) != original_row_count:
        raise AssertionError("The raw input table was unexpectedly changed.")
    if original_row_count - duplicate_count - flagged_count != len(attacks):
        raise AssertionError("Final row count does not match the cleaning steps.")

    return attacks, pd.DataFrame(report_rows)


def label_strategies(attacks: pd.DataFrame, show_samples: bool = True) -> pd.DataFrame:
    """Add multi-label rule-based strategy flags and print short review samples."""
    attack_text = attacks["attacker_input"].fillna("").astype(str)
    strategy_columns = []

    # An attack can match multiple detectors because strategies can be combined.
    for strategy, pattern in STRATEGY_PATTERNS.items():
        column = f"strategy_{strategy}"
        if strategy == "repeated_character_prefix":
            # Compare the first three characters directly to avoid regex capture groups.
            first_three = attack_text.str.slice(0, 3)
            attacks[column] = (
                first_three.str.len().eq(3)
                & first_three.str[0].eq(first_three.str[1])
                & first_three.str[1].eq(first_three.str[2])
            )
        else:
            attacks[column] = attack_text.str.contains(pattern, na=False)
        strategy_columns.append(column)

    # The other bucket means no listed detector matched this attack.
    has_strategy = attacks[strategy_columns].any(axis=1)
    attacks["strategy_other"] = ~has_strategy
    strategy_columns.append("strategy_other")

    total_attacks = len(attacks)
    print(f"\nStage 2 attacks labeled: {total_attacks:,}")
    report_rows = []

    for column in strategy_columns:
        strategy = column.removeprefix("strategy_")
        matching = attacks.loc[attacks[column]]
        covered = len(matching)
        percent = (100 * covered / total_attacks) if total_attacks else 0.0
        report_rows.append(
            {
                "strategy": strategy,
                "attacks_covered": covered,
                "percent_of_attacks": percent,
            }
        )
        print(f"{strategy}: {covered:,} attacks ({percent:.2f}%)")

        if show_samples:
            # Only print at most 20 randomly selected excerpts, capped at 150 chars.
            sample = matching["attacker_input"].sample(
                n=min(20, covered), random_state=RANDOM_SEED
            )
            print(f"  Validation sample size: {len(sample)}")
            for sample_text in sample:
                short_text = re.sub(r"\s+", " ", str(sample_text))[:150]
                print(f"  - {short_text}")

    unlabeled_count = int(attacks["strategy_other"].sum())
    unlabeled_percent = (100 * unlabeled_count / total_attacks) if total_attacks else 0.0
    print(f"Unlabeled attacks: {unlabeled_count:,} ({unlabeled_percent:.2f}%)")

    # Verify every attack is either covered or assigned to the other bucket.
    if not (attacks[strategy_columns].any(axis=1).all() if total_attacks else True):
        raise AssertionError("At least one attack has no strategy flag, including other.")
    attacks.attrs["stage2_report"] = pd.DataFrame(report_rows)
    return attacks


def build_consecutive_pairs(
    attacks: pd.DataFrame, session_gap_minutes: int = SESSION_GAP_MINUTES
) -> pd.DataFrame:
    """Create within-session consecutive pairs for each attacking player."""
    player_column = "attacker_id_anonymized"
    strategy_columns = [
        f"strategy_{strategy}" for strategy in STRATEGY_PATTERNS
    ]
    # "Other" combines unrelated tactics, so it is not treated as a strategy here.
    player_attacks = attacks.loc[attacks[player_column].notna()].copy()
    missing_player_count = len(attacks) - len(player_attacks)
    print(f"Attacks with player IDs: {len(player_attacks):,}; missing player IDs excluded: {missing_player_count:,}")
    player_attacks = player_attacks.sort_values(
        [player_column, "timestamp"], kind="stable"
    )
    grouped = player_attacks.groupby(player_column, sort=False)

    # Shifting by player pairs each attack only with that player's previous attack.
    previous_timestamp = grouped["timestamp"].shift()
    previous_success = grouped["success"].shift()
    time_gap = player_attacks["timestamp"] - previous_timestamp
    inside_session = previous_timestamp.notna() & time_gap.le(
        pd.Timedelta(minutes=session_gap_minutes)
    )

    # A repeat means the two attacks share at least one non-other label.
    same_strategy = pd.Series(False, index=player_attacks.index)
    for column in strategy_columns:
        previous_label = grouped[column].shift().fillna(False).astype(bool)
        same_strategy |= player_attacks[column] & previous_label

    pairs = pd.DataFrame(
        {
            player_column: player_attacks.loc[inside_session, player_column],
            "timestamp": player_attacks.loc[inside_session, "timestamp"],
            "previous_success": previous_success.loc[inside_session].astype(bool),
            "same_strategy": same_strategy.loc[inside_session].astype(bool),
        }
    ).reset_index(drop=True)

    player_count = pairs[player_column].nunique()
    print(
        f"Consecutive pairs within {session_gap_minutes}-minute sessions: "
        f"{len(pairs):,}; players contributing pairs: {player_count:,}"
    )
    print(
        f"Pairs after success: {int(pairs['previous_success'].sum()):,}; "
        f"pairs after failure: {int((~pairs['previous_success']).sum()):,}"
    )
    if pairs.empty:
        raise ValueError("No within-session consecutive attack pairs were found.")
    return pairs


def chance_repeat_rate(attacks: pd.DataFrame) -> float:
    """Estimate repeat chance from independent draws of observed label sets."""
    strategy_columns = [
        f"strategy_{strategy}" for strategy in STRATEGY_PATTERNS
    ]
    labels = attacks[strategy_columns].to_numpy(dtype=np.int64)
    # Encode each multi-label combination as a bit mask, including unlabeled rows.
    bit_weights = 1 << np.arange(len(strategy_columns), dtype=np.int64)
    codes = labels @ bit_weights
    unique_codes, counts = np.unique(codes, return_counts=True)

    # The independent-draw baseline preserves observed strategy co-occurrence.
    overlap = (unique_codes[:, None] & unique_codes[None, :]) != 0
    weighted_pairs = counts[:, None] * counts[None, :] * overlap
    return float(weighted_pairs.sum() / (len(attacks) ** 2))


def bootstrap_repeat_rates(
    pairs: pd.DataFrame, replicates: int = BOOTSTRAP_REPLICATES
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    """Bootstrap players as clusters and return rate and difference intervals."""
    player_column = "attacker_id_anonymized"
    player_pairs = pairs.copy()
    player_pairs["success_pair"] = player_pairs["previous_success"]
    player_pairs["success_repeat"] = (
        player_pairs["previous_success"] & player_pairs["same_strategy"]
    )
    player_pairs["failure_pair"] = ~player_pairs["previous_success"]
    player_pairs["failure_repeat"] = (
        ~player_pairs["previous_success"] & player_pairs["same_strategy"]
    )

    # Each row of this table is one player, so resampling rows resamples players.
    player_totals = player_pairs.groupby(player_column)[
        ["success_repeat", "success_pair", "failure_repeat", "failure_pair"]
    ].sum()
    values = player_totals.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_players = rng.integers(
        0, len(values), size=(replicates, len(values))
    )
    sampled_totals = values[sampled_players].sum(axis=1)

    success_denominator = sampled_totals[:, 1]
    failure_denominator = sampled_totals[:, 3]
    success_rates = np.divide(
        sampled_totals[:, 0], success_denominator,
        out=np.full(replicates, np.nan), where=success_denominator > 0,
    )
    failure_rates = np.divide(
        sampled_totals[:, 2], failure_denominator,
        out=np.full(replicates, np.nan), where=failure_denominator > 0,
    )
    differences = success_rates - failure_rates

    intervals = {
        "success": tuple(np.nanquantile(success_rates, [0.025, 0.975])),
        "failure": tuple(np.nanquantile(failure_rates, [0.025, 0.975])),
    }
    difference_interval = tuple(np.nanquantile(differences, [0.025, 0.975]))
    return intervals, difference_interval


def analyze_win_stay_lose_shift(
    attacks: pd.DataFrame,
    session_gap_minutes: int = SESSION_GAP_MINUTES,
) -> pd.DataFrame:
    """Compute repeat rates, player-bootstrap intervals, and chance baseline."""
    pairs = build_consecutive_pairs(attacks, session_gap_minutes)
    chance_rate = chance_repeat_rate(attacks)
    intervals, difference_interval = bootstrap_repeat_rates(pairs)
    rows = []

    for outcome, condition in (("success", pairs["previous_success"]), ("failure", ~pairs["previous_success"])):
        selected = pairs.loc[condition]
        pair_count = len(selected)
        repeat_count = int(selected["same_strategy"].sum())
        repeat_rate = repeat_count / pair_count if pair_count else np.nan
        low, high = intervals[outcome]
        rows.append(
            {
                "measure": "repeat_rate",
                "previous_outcome": outcome,
                "pair_count": pair_count,
                "player_count": selected["attacker_id_anonymized"].nunique(),
                "repeat_count": repeat_count,
                "estimate": repeat_rate,
                "ci95_low": low,
                "ci95_high": high,
                "chance_baseline": chance_rate,
            }
        )
        print(
            f"P(same strategy | previous {outcome}): {repeat_count:,}/{pair_count:,} "
            f"= {repeat_rate:.4f}; 95% player-bootstrap CI [{low:.4f}, {high:.4f}]"
        )

    success_rate = rows[0]["estimate"]
    failure_rate = rows[1]["estimate"]
    difference = success_rate - failure_rate
    diff_low, diff_high = difference_interval
    rows.append(
        {
            "measure": "success_minus_failure_difference",
            "previous_outcome": "success minus failure",
            "pair_count": len(pairs),
            "player_count": pairs["attacker_id_anonymized"].nunique(),
            "repeat_count": np.nan,
            "estimate": difference,
            "ci95_low": diff_low,
            "ci95_high": diff_high,
            "chance_baseline": 0.0,
        }
    )
    print(
        f"Success-minus-failure difference: {difference:.4f}; "
        f"95% player-bootstrap CI [{diff_low:.4f}, {diff_high:.4f}]"
    )
    print(f"Chance baseline (independent observed label sets): {chance_rate:.4f}")

    # Draw only the two observed conditional rates and the common chance baseline.
    fig, axis = plt.subplots(figsize=(7, 5))
    estimates = [rows[0]["estimate"], rows[1]["estimate"]]
    lower_errors = [estimates[i] - rows[i]["ci95_low"] for i in range(2)]
    upper_errors = [rows[i]["ci95_high"] - estimates[i] for i in range(2)]
    axis.bar(["After success", "After failure"], estimates, color=["#2f7f73", "#d1854b"])
    axis.errorbar(
        [0, 1], estimates, yerr=[lower_errors, upper_errors],
        fmt="none", ecolor="#202020", capsize=5,
    )
    axis.axhline(chance_rate, color="#4a6472", linestyle="--", label="Chance baseline")
    axis.set_ylabel("Probability next attack shares a strategy")
    axis.set_title(f"Strategy repetition (session gap: {session_gap_minutes} min)")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    fig.tight_layout()
    chart_path = OUTPUT_DIR / "stage3_win_stay_lose_shift.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    print(f"Chart saved to: {chart_path.relative_to(PROJECT_DIR)}")
    return pd.DataFrame(rows)


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a 95% Wilson interval for a binomial proportion."""
    if trials == 0:
        return np.nan, np.nan
    interval = binomtest(successes, trials).proportion_ci(method="wilson")
    return float(interval.low), float(interval.high)


def newcombe_difference_interval(
    first_rate: float,
    first_interval: tuple[float, float],
    second_rate: float,
    second_interval: tuple[float, float],
) -> tuple[float, float]:
    """Combine Wilson intervals into a Newcombe interval for a rate difference."""
    difference = first_rate - second_rate
    low = difference - np.sqrt(
        (first_rate - first_interval[0]) ** 2
        + (second_interval[1] - second_rate) ** 2
    )
    high = difference + np.sqrt(
        (first_interval[1] - first_rate) ** 2
        + (second_rate - second_interval[0]) ** 2
    )
    return float(low), float(high)


def benjamini_hochberg(p_values: list[float]) -> np.ndarray:
    """Adjust a family of p-values to control the false discovery rate."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running_minimum = 1.0
    for rank in range(len(order) - 1, -1, -1):
        index = order[rank]
        running_minimum = min(
            running_minimum, values[index] * len(values) / (rank + 1)
        )
        adjusted[index] = running_minimum
    return adjusted


def analyze_session_gap_sensitivity(
    attacks: pd.DataFrame, session_gaps: tuple[int, ...] = (30, 60, 120)
) -> pd.DataFrame:
    """Compare win-stay/lose-shift estimates across session-gap choices."""
    rows = []
    for gap_minutes in session_gaps:
        pairs = build_consecutive_pairs(attacks, gap_minutes)
        intervals, difference_interval = bootstrap_repeat_rates(pairs)
        success_pairs = pairs.loc[pairs["previous_success"]]
        failure_pairs = pairs.loc[~pairs["previous_success"]]
        success_rate = float(success_pairs["same_strategy"].mean())
        failure_rate = float(failure_pairs["same_strategy"].mean())
        difference = success_rate - failure_rate
        row = {
            "session_gap_minutes": gap_minutes,
            "pair_count": len(pairs),
            "player_count": pairs["attacker_id_anonymized"].nunique(),
            "success_pair_count": len(success_pairs),
            "success_repeat_count": int(success_pairs["same_strategy"].sum()),
            "success_repeat_rate": success_rate,
            "success_ci95_low": intervals["success"][0],
            "success_ci95_high": intervals["success"][1],
            "failure_pair_count": len(failure_pairs),
            "failure_repeat_count": int(failure_pairs["same_strategy"].sum()),
            "failure_repeat_rate": failure_rate,
            "failure_ci95_low": intervals["failure"][0],
            "failure_ci95_high": intervals["failure"][1],
            "difference": difference,
            "difference_ci95_low": difference_interval[0],
            "difference_ci95_high": difference_interval[1],
        }
        rows.append(row)
        print(
            f"Session-gap sensitivity, {gap_minutes} minutes: "
            f"{len(pairs):,} pairs; after success {success_rate:.4f}; "
            f"after failure {failure_rate:.4f}; difference {difference:.4f} "
            f"(95% player-bootstrap CI "
            f"[{difference_interval[0]:.4f}, {difference_interval[1]:.4f}])"
        )
    return pd.DataFrame(rows)


def analyze_calendar_adjusted_persistence(attacks: pd.DataFrame) -> pd.DataFrame:
    """Estimate repetition after success controlling for UTC calendar week."""
    pairs = build_consecutive_pairs(attacks, SESSION_GAP_MINUTES)
    pairs["previous_success"] = pairs["previous_success"].astype(int)
    pairs["same_strategy"] = pairs["same_strategy"].astype(int)
    pairs["calendar_week"] = pairs["timestamp"].dt.strftime("%G-W%V")

    # Calendar-week fixed effects control shared changes in strategy popularity.
    # Cluster-robust errors account for multiple pairs from the same player.
    model = smf.glm(
        "same_strategy ~ previous_success + C(calendar_week)",
        data=pairs,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": pairs["attacker_id_anonymized"]})
    low, high = model.conf_int().loc["previous_success"]
    coefficient = float(model.params["previous_success"])
    result = pd.DataFrame(
        [
            {
                "session_gap_minutes": SESSION_GAP_MINUTES,
                "calendar_weeks": pairs["calendar_week"].nunique(),
                "pair_count": len(pairs),
                "player_clusters": pairs["attacker_id_anonymized"].nunique(),
                "log_odds_coefficient": coefficient,
                "odds_ratio": float(np.exp(coefficient)),
                "odds_ratio_ci95_low": float(np.exp(low)),
                "odds_ratio_ci95_high": float(np.exp(high)),
                "p_value": float(model.pvalues["previous_success"]),
            }
        ]
    )
    print(
        f"Calendar-week-adjusted repetition: OR={result.iloc[0]['odds_ratio']:.3f}; "
        f"95% CI [{result.iloc[0]['odds_ratio_ci95_low']:.3f}, "
        f"{result.iloc[0]['odds_ratio_ci95_high']:.3f}], "
        f"player-clustered p={result.iloc[0]['p_value']:.4g}; "
        f"weeks={result.iloc[0]['calendar_weeks']}, pairs={len(pairs):,}"
    )
    return result


def analyze_social_learning(
    attacks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare strategy adoption by prior exposure and calculate weekly popularity."""
    attacker_column = "attacker_id_anonymized"
    defender_column = "defender_id_anonymized"
    # Convert to arrays first so pandas metadata on the columns cannot affect concat.
    player_ids = pd.Index(
        np.unique(
            np.concatenate(
                [
                    attacks[attacker_column].dropna().to_numpy(),
                    attacks[defender_column].dropna().to_numpy(),
                ]
            )
        )
    )
    print(f"Players considered (attacker or defender): {len(player_ids):,}")

    # Self-attacks do not count as being exposed to another player's strategy.
    incoming_attacks = attacks.loc[~attacks["is_self_attack"]]
    print(
        f"Incoming exposure events considered: {len(incoming_attacks):,}; "
        f"self-attacks excluded from exposure: {int(attacks['is_self_attack'].sum()):,}"
    )

    adoption_rows = []
    observation_end = attacks["timestamp"].max()

    for strategy in STRATEGY_PATTERNS:
        strategy_column = f"strategy_{strategy}"

        # First outgoing use is the player's adoption time for this strategy.
        first_use = (
            attacks.loc[attacks[strategy_column]]
            .groupby(attacker_column)["timestamp"]
            .min()
            .reindex(player_ids)
        )
        adopted = first_use.notna()

        # First incoming use by another attacker is that player's exposure time.
        first_exposure = (
            incoming_attacks.loc[incoming_attacks[strategy_column]]
            .groupby(defender_column)["timestamp"]
            .min()
            .reindex(player_ids)
        )

        # For adopters, exposure must be strictly earlier than first use.
        # For non-adopters, exposure is measured through the end of observation.
        prior_exposure = (first_exposure < first_use) & adopted
        prior_exposure |= (~adopted) & first_exposure.le(observation_end)

        exposed_mask = prior_exposure
        unexposed_mask = ~prior_exposure
        exposed_n = int(exposed_mask.sum())
        unexposed_n = int(unexposed_mask.sum())
        exposed_adopters = int((exposed_mask & adopted).sum())
        unexposed_adopters = int((unexposed_mask & adopted).sum())
        exposed_rate = exposed_adopters / exposed_n if exposed_n else np.nan
        unexposed_rate = unexposed_adopters / unexposed_n if unexposed_n else np.nan
        exposed_interval = wilson_interval(exposed_adopters, exposed_n)
        unexposed_interval = wilson_interval(unexposed_adopters, unexposed_n)

        if exposed_n and unexposed_n:
            risk_difference = exposed_rate - unexposed_rate
            difference_interval = newcombe_difference_interval(
                exposed_rate, exposed_interval, unexposed_rate, unexposed_interval
            )
            fisher_p = float(
                fisher_exact(
                    [
                        [exposed_adopters, exposed_n - exposed_adopters],
                        [unexposed_adopters, unexposed_n - unexposed_adopters],
                    ]
                ).pvalue
            )
        else:
            risk_difference = np.nan
            difference_interval = (np.nan, np.nan)
            fisher_p = np.nan

        adoption_rows.append(
            {
                "strategy": strategy,
                "exposed_adopters": exposed_adopters,
                "exposed_players": exposed_n,
                "exposed_adoption_rate": exposed_rate,
                "exposed_ci95_low": exposed_interval[0],
                "exposed_ci95_high": exposed_interval[1],
                "unexposed_adopters": unexposed_adopters,
                "unexposed_players": unexposed_n,
                "unexposed_adoption_rate": unexposed_rate,
                "unexposed_ci95_low": unexposed_interval[0],
                "unexposed_ci95_high": unexposed_interval[1],
                "risk_difference": risk_difference,
                "risk_difference_ci95_low": difference_interval[0],
                "risk_difference_ci95_high": difference_interval[1],
                "fisher_exact_p": fisher_p,
            }
        )
        print(
            f"{strategy}: exposed {exposed_adopters:,}/{exposed_n:,} "
            f"({exposed_rate:.2%}; 95% Wilson CI "
            f"[{exposed_interval[0]:.2%}, {exposed_interval[1]:.2%}]); "
            f"unexposed {unexposed_adopters:,}/{unexposed_n:,} "
            f"({unexposed_rate:.2%}; 95% Wilson CI "
            f"[{unexposed_interval[0]:.2%}, {unexposed_interval[1]:.2%}]); "
            f"risk difference {risk_difference:.2%}; Fisher p={fisher_p:.4g}"
        )

        # Confirm each player belongs to exactly one exposure group per strategy.
        if int(prior_exposure.sum()) + int((~prior_exposure).sum()) != len(player_ids):
            raise AssertionError(f"Exposure grouping failed for strategy: {strategy}")

    adoption_report = pd.DataFrame(adoption_rows)
    adoption_report["fdr_q_value"] = benjamini_hochberg(
        adoption_report["fisher_exact_p"].fillna(1.0).tolist()
    )

    # Weekly popularity is the share of attacks carrying each rule-based label.
    # Multi-label percentages can overlap and therefore need not sum to 100%.
    weekly_source = attacks.set_index("timestamp").sort_index()
    weekly_aggregate = weekly_source.resample("W-SUN").agg(
        {"attack_id": "count", **{f"strategy_{s}": "sum" for s in STRATEGY_PATTERNS}}
    )
    weekly_rows = []
    for week, row in weekly_aggregate.iterrows():
        total_attacks = int(row["attack_id"])
        for strategy in STRATEGY_PATTERNS:
            strategy_count = int(row[f"strategy_{strategy}"])
            weekly_rows.append(
                {
                    "week_ending": week,
                    "strategy": strategy,
                    "attack_count": strategy_count,
                    "total_attacks": total_attacks,
                    "percent_of_attacks": (
                        100 * strategy_count / total_attacks if total_attacks else 0.0
                    ),
                }
            )
    weekly_report = pd.DataFrame(weekly_rows)
    print(
        f"Weekly popularity observations: {weekly_aggregate.shape[0]:,} weeks; "
        f"attacks included: {len(attacks):,}"
    )

    # Plot exposed versus unexposed adoption rates for each strategy.
    positions = np.arange(len(adoption_report))
    bar_width = 0.38
    fig, rate_axis = plt.subplots(figsize=(11, 6))
    rate_axis.bar(
        positions - bar_width / 2,
        adoption_report["exposed_adoption_rate"],
        bar_width,
        color="#2f7f73",
        label="Exposed",
    )
    rate_axis.bar(
        positions + bar_width / 2,
        adoption_report["unexposed_adoption_rate"],
        bar_width,
        color="#d1854b",
        label="Unexposed",
    )
    for offset, estimate_column, low_column, high_column in (
        (-bar_width / 2, "exposed_adoption_rate", "exposed_ci95_low", "exposed_ci95_high"),
        (bar_width / 2, "unexposed_adoption_rate", "unexposed_ci95_low", "unexposed_ci95_high"),
    ):
        estimates = adoption_report[estimate_column].to_numpy()
        errors = np.vstack(
            [
                estimates - adoption_report[low_column].to_numpy(),
                adoption_report[high_column].to_numpy() - estimates,
            ]
        )
        rate_axis.errorbar(
            positions + offset, estimates, yerr=errors,
            fmt="none", ecolor="#202020", capsize=2, linewidth=0.8,
        )
    rate_axis.set_xticks(positions)
    rate_axis.set_xticklabels(adoption_report["strategy"], rotation=35, ha="right")
    rate_axis.set_ylabel("Adoption rate")
    rate_axis.set_xlabel("")
    rate_axis.set_title("Strategy adoption by prior exposure")
    rate_axis.set_ylim(bottom=0)
    rate_axis.legend(title="Prior exposure", frameon=False)
    fig.tight_layout()
    adoption_chart_path = OUTPUT_DIR / "stage4_adoption_rates.png"
    fig.savefig(adoption_chart_path, dpi=160)
    plt.close(fig)

    # Plot weekly share as a percent so volume changes do not dominate the chart.
    weekly_pivot = weekly_report.pivot(
        index="week_ending", columns="strategy", values="percent_of_attacks"
    )
    weekly_axis = weekly_pivot.plot(figsize=(12, 6), marker="o", markersize=3)
    weekly_axis.set_ylabel("Attacks with strategy label (%)")
    weekly_axis.set_xlabel("Week ending (UTC)")
    weekly_axis.set_title("Weekly strategy popularity")
    weekly_axis.legend(title="Strategy", frameon=False, ncol=2)
    weekly_axis.figure.tight_layout()
    weekly_chart_path = OUTPUT_DIR / "stage4_weekly_strategy_popularity.png"
    weekly_axis.figure.savefig(weekly_chart_path, dpi=160)
    plt.close(weekly_axis.figure)
    print(f"Adoption chart saved to: {adoption_chart_path.relative_to(PROJECT_DIR)}")
    print(f"Weekly chart saved to: {weekly_chart_path.relative_to(PROJECT_DIR)}")

    return adoption_report, weekly_report


def analyze_pre_exposure_cohort(attacks: pd.DataFrame) -> pd.DataFrame:
    """Test later strategy use among players exposed before their first attack."""
    attacker_column = "attacker_id_anonymized"
    defender_column = "defender_id_anonymized"
    players = pd.Index(attacks[attacker_column].dropna().unique())
    first_outgoing = attacks.groupby(attacker_column)["timestamp"].min().reindex(players)
    incoming = attacks.loc[~attacks["is_self_attack"]]
    first_incoming = (
        incoming.groupby(defender_column)["timestamp"].min().reindex(players)
    )
    exposed_before_first_attack = (first_incoming < first_outgoing).fillna(False)
    exposed_count = int(exposed_before_first_attack.sum())
    unexposed_count = len(players) - exposed_count
    print(
        f"Pre-exposure cohort: {exposed_count:,} players received any non-self "
        f"attack before their first outgoing attack; "
        f"{unexposed_count:,} did not."
    )

    # Count use only after the first outgoing attack, so pre-baseline use cannot
    # be mistaken for a later response to exposure.
    later_attacks = attacks.loc[
        attacks["timestamp"].gt(attacks[attacker_column].map(first_outgoing))
    ]
    rows = []
    for strategy in STRATEGY_PATTERNS:
        strategy_column = f"strategy_{strategy}"
        later_use = (
            later_attacks.groupby(attacker_column)[strategy_column]
            .any()
            .reindex(players)
            .fillna(False)
            .astype(bool)
        )
        exposed_adopters = int((exposed_before_first_attack & later_use).sum())
        unexposed_adopters = int((~exposed_before_first_attack & later_use).sum())
        exposed_interval = wilson_interval(exposed_adopters, exposed_count)
        unexposed_interval = wilson_interval(unexposed_adopters, unexposed_count)
        exposed_rate = exposed_adopters / exposed_count if exposed_count else np.nan
        unexposed_rate = unexposed_adopters / unexposed_count if unexposed_count else np.nan
        risk_difference = exposed_rate - unexposed_rate
        difference_interval = newcombe_difference_interval(
            exposed_rate, exposed_interval, unexposed_rate, unexposed_interval
        )
        fisher_p = float(
            fisher_exact(
                [
                    [exposed_adopters, exposed_count - exposed_adopters],
                    [unexposed_adopters, unexposed_count - unexposed_adopters],
                ]
            ).pvalue
        )
        rows.append(
            {
                "strategy": strategy,
                "players_exposed_before_first_attack": exposed_count,
                "players_without_prior_exposure": unexposed_count,
                "exposed_later_users": exposed_adopters,
                "exposed_later_use_rate": exposed_rate,
                "exposed_ci95_low": exposed_interval[0],
                "exposed_ci95_high": exposed_interval[1],
                "unexposed_later_users": unexposed_adopters,
                "unexposed_later_use_rate": unexposed_rate,
                "unexposed_ci95_low": unexposed_interval[0],
                "unexposed_ci95_high": unexposed_interval[1],
                "risk_difference": risk_difference,
                "risk_difference_ci95_low": difference_interval[0],
                "risk_difference_ci95_high": difference_interval[1],
                "fisher_exact_p": fisher_p,
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = benjamini_hochberg(result["fisher_exact_p"].tolist())
    print(
        f"Later attacks in pre-exposure cohort analysis: {len(later_attacks):,}; "
        f"players with follow-up attacks: {later_attacks[attacker_column].nunique():,}"
    )
    return result


def main() -> None:
    """Run Stages 1 to 4; save aggregate tables and charts, never attack text."""
    print(f"Random seed: {RANDOM_SEED}")

    # Read the compressed JSON Lines file; each line is one attack record.
    raw_attacks = pd.read_json(ATTACKS_FILE, lines=True, compression="bz2")
    cleaned_attacks, cleaning_report = clean_attacks(raw_attacks)
    labeled_attacks = label_strategies(cleaned_attacks, show_samples=False)
    win_stay_report = analyze_win_stay_lose_shift(labeled_attacks)
    session_sensitivity = analyze_session_gap_sensitivity(labeled_attacks)
    calendar_adjusted_report = analyze_calendar_adjusted_persistence(labeled_attacks)
    adoption_report, weekly_report = analyze_social_learning(labeled_attacks)
    pre_exposure_report = analyze_pre_exposure_cohort(labeled_attacks)

    # Save only step names and row counts, not prompts or model responses.
    OUTPUT_DIR.mkdir(exist_ok=True)
    cleaning_report_path = OUTPUT_DIR / "stage1_cleaning_summary.csv"
    cleaning_report.to_csv(cleaning_report_path, index=False)
    strategy_report_path = OUTPUT_DIR / "stage2_strategy_counts.csv"
    labeled_attacks.attrs["stage2_report"].to_csv(strategy_report_path, index=False)
    stage3_report_path = OUTPUT_DIR / "stage3_win_stay_lose_shift.csv"
    win_stay_report.to_csv(stage3_report_path, index=False)
    adoption_report_path = OUTPUT_DIR / "stage4_social_learning.csv"
    adoption_report.to_csv(adoption_report_path, index=False)
    weekly_report_path = OUTPUT_DIR / "stage4_weekly_strategy_popularity.csv"
    weekly_report.to_csv(weekly_report_path, index=False)
    session_sensitivity_path = OUTPUT_DIR / "stage3_session_sensitivity.csv"
    session_sensitivity.to_csv(session_sensitivity_path, index=False)
    calendar_adjusted_path = OUTPUT_DIR / "stage3_calendar_adjusted.csv"
    calendar_adjusted_report.to_csv(calendar_adjusted_path, index=False)
    pre_exposure_path = OUTPUT_DIR / "stage4_pre_exposure_sensitivity.csv"
    pre_exposure_report.to_csv(pre_exposure_path, index=False)
    print(f"Stage 1 aggregate counts saved to: {cleaning_report_path.relative_to(PROJECT_DIR)}")
    print(f"Stage 2 aggregate counts saved to: {strategy_report_path.relative_to(PROJECT_DIR)}")
    print(f"Stage 3 aggregate results saved to: {stage3_report_path.relative_to(PROJECT_DIR)}")
    print(f"Stage 4 adoption summary saved to: {adoption_report_path.relative_to(PROJECT_DIR)}")
    print(f"Stage 4 weekly summary saved to: {weekly_report_path.relative_to(PROJECT_DIR)}")
    print(f"Stage 3 session sensitivity saved to: {session_sensitivity_path.relative_to(PROJECT_DIR)}")
    print(f"Stage 3 calendar-adjusted results saved to: {calendar_adjusted_path.relative_to(PROJECT_DIR)}")
    print(f"Stage 4 pre-exposure sensitivity saved to: {pre_exposure_path.relative_to(PROJECT_DIR)}")
    print(f"Final cleaned attack count: {len(labeled_attacks):,}")


if __name__ == "__main__":
    main()