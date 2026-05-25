#!/usr/bin/env python3
"""Aggregate all experiment eval results into a summary CSV and print summary."""
import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict

BASE = Path("/root/provenance_weight_training")
OUTPUT = BASE / "output"
MODELS = OUTPUT / "models"
RESULTS_DIR = BASE / "results"


def parse_exp_id(key):
    """Parse experiment key like 'grid_b0_5_seed42/final' -> dict of attributes."""
    key = key.split("/")[0]

    seed_match = re.search(r"seed(\d+)", key)
    seed = int(seed_match.group(1)) if seed_match else None

    b_value = None
    tau = None
    n_min = None
    batch_size = None
    method = "unknown"
    variant = ""

    if key.startswith("uniform"):
        method = "uniform"
    elif key.startswith("grid_b"):
        method = "grid"
        b_match = re.search(r"grid_b(\d+_\d+)", key)
        if b_match:
            b_value = float(b_match.group(1).replace("_", "."))
    elif key.startswith("adaptive_fix"):
        method = "adaptive_fix"
    elif key.startswith("adaptive"):
        method = "adaptive"
    elif key.startswith("diagnostic_contamination"):
        c_match = re.search(r"contamination(\d+)", key)
        method = "diagnostic"
        variant = f"c{c_match.group(1)}" if c_match else ""
    elif key.startswith("ess_only"):
        method = "ess_only"
    elif key.startswith("golden_ratio"):
        method = "golden_ratio"
    elif key.startswith("nmin"):
        method = "adaptive"
        n_match = re.search(r"nmin(\d+)", key)
        n_min = int(n_match.group(1)) if n_match else None
        variant = f"nmin{n_min}"
    elif key.startswith("tau_"):
        method = "adaptive"
        t_match = re.search(r"tau_?(\d+\.?\d*)", key)
        tau = float(t_match.group(1)) if t_match else None
        variant = f"tau{tau}"
    elif key.startswith("batchsize"):
        method = "adaptive"
        bs_match = re.search(r"batchsize(\d+)", key)
        batch_size = int(bs_match.group(1)) if bs_match else None
        variant = f"bs{batch_size}"
    else:
        method = key

    return {
        "exp_id": key,
        "method": method,
        "variant": variant,
        "seed": seed,
        "b_value": b_value,
        "tau": tau,
        "n_min": n_min,
        "batch_size": batch_size,
    }


def load_eval_json(path):
    """Load a JSON eval file. Returns list of (exp_key, ppl, loss) tuples."""
    results = []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return results

    for key, val in data.items():
        if isinstance(val, dict):
            ppl = val.get("perplexity")
            loss = val.get("loss")
            if ppl is not None:
                results.append((key, ppl, loss, str(path)))
            elif isinstance(val, dict) and "perplexity" in val:
                inner = val
                ppl = inner.get("perplexity")
                loss = inner.get("loss")
                if ppl is not None:
                    results.append((key, ppl, loss, str(path)))
    return results


def load_nested_eval_json(path):
    """Load eval JSON with nested {perplexity: {perplexity: ..., loss: ...}} format."""
    results = []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return results

    if "perplexity" in data and isinstance(data["perplexity"], dict):
        inner = data["perplexity"]
        ppl = inner.get("perplexity")
        loss = inner.get("loss")
        exp_key = path.parent.name + "/final"
        if ppl is not None:
            results.append((exp_key, ppl, loss, str(path)))
    else:
        results.extend(load_eval_json(path))
    return results


def extract_ppl_from_trainer_state(exp_dir):
    """Fallback: last eval_loss from trainer_state.json -> exp(eval_loss)."""
    candidates = sorted(exp_dir.glob("checkpoint-*/trainer_state.json"))
    if (exp_dir / "final" / "trainer_state.json").exists():
        candidates.append(exp_dir / "final" / "trainer_state.json")
    if not candidates:
        return None, None
    state_path = candidates[-1]
    try:
        state = json.loads(state_path.read_text())
        log_history = state.get("log_history", [])
        eval_entries = [e for e in log_history if "eval_loss" in e]
        if not eval_entries:
            return None, None
        last_eval_loss = eval_entries[-1]["eval_loss"]
        return math.exp(last_eval_loss), last_eval_loss
    except (json.JSONDecodeError, KeyError, OverflowError):
        return None, None


def get_training_status(exp_dir):
    """Check if experiment is done, running, or missing."""
    if not exp_dir.exists():
        return "NOT_FOUND"
    if (exp_dir / "final").exists():
        return "DONE"
    checkpoints = sorted(exp_dir.glob("checkpoint-*"))
    if checkpoints:
        last_ckpt = int(checkpoints[-1].name.split("-")[1])
        return f"RUNNING(step={last_ckpt})"
    if (exp_dir / "config.json").exists():
        return "STARTED"
    return "NOT_FOUND"


def collect_all_eval_results():
    """Scan all eval data sources and return deduplicated results."""
    all_results = {}

    # Source 1: output/eval_perplexity_results.json
    for item in load_eval_json(OUTPUT / "eval_perplexity_results.json"):
        key, ppl, loss, src = item
        all_results[key.split("/")[0]] = (ppl, loss, "eval_perplexity_results")

    # Source 2: output/eval_results/*_eval.json
    eval_results_dir = OUTPUT / "eval_results"
    if eval_results_dir.exists():
        for f in sorted(eval_results_dir.glob("*_eval.json")):
            for key, ppl, loss, src in load_eval_json(f):
                exp_id = key.split("/")[0]
                if exp_id not in all_results:
                    all_results[exp_id] = (ppl, loss, f"eval_results/{f.name}")

    # Source 3: output/eval/posthoc_eval*.json
    eval_dir = OUTPUT / "eval"
    if eval_dir.exists():
        for f in sorted(eval_dir.glob("posthoc_eval*.json")):
            for key, ppl, loss, src in load_eval_json(f):
                exp_id = key.split("/")[0]
                if exp_id not in all_results:
                    all_results[exp_id] = (ppl, loss, f"eval/{f.name}")

    # Source 4: output/eval_diagnostic_*.json
    for f in sorted(OUTPUT.glob("eval_diagnostic_*.json")):
        for key, ppl, loss, src in load_eval_json(f):
            exp_id = key.split("/")[0]
            if exp_id not in all_results:
                all_results[exp_id] = (ppl, loss, f.name)

    # Source 5: output/eval_grid_*.json (standalone)
    for f in sorted(OUTPUT.glob("eval_grid_*.json")):
        for key, ppl, loss, src in load_eval_json(f):
            exp_id = key.split("/")[0]
            if exp_id not in all_results:
                all_results[exp_id] = (ppl, loss, f.name)

    # Source 6: output/models/*/eval_results.json
    if MODELS.exists():
        for exp_dir in sorted(MODELS.iterdir()):
            if not exp_dir.is_dir():
                continue
            er = exp_dir / "eval_results.json"
            if er.exists():
                for key, ppl, loss, src in load_nested_eval_json(er):
                    exp_id = key.split("/")[0]
                    if exp_id not in all_results:
                        all_results[exp_id] = (ppl, loss, f"models/{exp_dir.name}/eval_results")

    # Source 7: output/eval/*/eval_results.json (nested format)
    if eval_dir.exists():
        for sub in sorted(eval_dir.iterdir()):
            if sub.is_dir() and (sub / "eval_results.json").exists():
                for key, ppl, loss, src in load_nested_eval_json(sub / "eval_results.json"):
                    exp_id = key.split("/")[0]
                    if exp_id not in all_results:
                        all_results[exp_id] = (ppl, loss, f"eval/{sub.name}/eval_results")

    return all_results


def mean_std(values):
    if not values:
        return None, None
    n = len(values)
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    return m, s


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all eval results
    eval_data = collect_all_eval_results()

    # Discover all experiment directories
    all_exp_dirs = set()
    if MODELS.exists():
        for d in MODELS.iterdir():
            if d.is_dir() and not d.name.endswith("_eval_tmp"):
                all_exp_dirs.add(d.name)

    # Add any experiment IDs from eval data not in models/
    for exp_id in eval_data:
        all_exp_dirs.add(exp_id)

    # Build rows
    rows = []
    for exp_id in sorted(all_exp_dirs):
        info = parse_exp_id(exp_id + "/final")
        exp_dir = MODELS / exp_id
        status = get_training_status(exp_dir)

        if exp_id in eval_data:
            ppl, loss, source = eval_data[exp_id]
            eval_source = source
        else:
            # Fallback: trainer_state.json
            ppl, loss = extract_ppl_from_trainer_state(exp_dir)
            eval_source = "trainer_state(fallback)" if ppl is not None else "none"

        rows.append({
            "exp_id": exp_id,
            "method": info["method"],
            "variant": info["variant"],
            "seed": info["seed"],
            "b_value": info["b_value"],
            "tau": info["tau"],
            "n_min": info["n_min"],
            "eval_ppl": ppl,
            "eval_loss": loss,
            "eval_source": eval_source,
            "status": status,
        })

    # Compute uniform baseline mean for relative_to_uniform
    uniform_ppls = [r["eval_ppl"] for r in rows if r["method"] == "uniform" and r["eval_ppl"] is not None]
    uniform_mean = sum(uniform_ppls) / len(uniform_ppls) if uniform_ppls else None

    # Add relative_to_uniform column
    for r in rows:
        if r["eval_ppl"] is not None and uniform_mean is not None:
            r["relative_to_uniform"] = (r["eval_ppl"] - uniform_mean) / uniform_mean * 100
        else:
            r["relative_to_uniform"] = None

    # Write CSV
    csv_path = RESULTS_DIR / "summary_table.csv"
    header = "exp_id,method,variant,seed,b_value,tau,n_min,eval_ppl,eval_loss,relative_to_uniform_pct,eval_source,status"
    with open(csv_path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            vals = [
                r["exp_id"],
                r["method"],
                r["variant"],
                str(r["seed"]) if r["seed"] is not None else "",
                f'{r["b_value"]:.1f}' if r["b_value"] is not None else "",
                f'{r["tau"]}' if r["tau"] is not None else "",
                str(r["n_min"]) if r["n_min"] is not None else "",
                f'{r["eval_ppl"]:.4f}' if r["eval_ppl"] is not None else "",
                f'{r["eval_loss"]:.6f}' if r["eval_loss"] is not None else "",
                f'{r["relative_to_uniform"]:.3f}' if r["relative_to_uniform"] is not None else "",
                r["eval_source"],
                r["status"],
            ]
            f.write(",".join(vals) + "\n")

    print(f"CSV written: {csv_path}")
    print(f"Total experiments: {len(rows)}")
    print(f"  With standalone eval: {sum(1 for r in rows if r['eval_source'] not in ('none', 'trainer_state(fallback)'))}")
    print(f"  With fallback eval: {sum(1 for r in rows if r['eval_source'] == 'trainer_state(fallback)')}")
    print(f"  No eval: {sum(1 for r in rows if r['eval_source'] == 'none')}")

    # ── Summary by method ──
    print("\n" + "=" * 90)
    print("  SUMMARY BY METHOD (standalone eval only)")
    print("=" * 90)

    if uniform_mean is not None:
        u_std = mean_std(uniform_ppls)[1]
        print(f"\n  Uniform baseline: {uniform_mean:.4f} ± {u_std:.4f} (n={len(uniform_ppls)})")

    method_groups = defaultdict(list)
    for r in rows:
        if r["eval_ppl"] is not None and r["eval_source"] not in ("trainer_state(fallback)",):
            key = r["method"]
            if r["variant"]:
                key += f"({r['variant']})"
            elif r["method"] == "grid" and r["b_value"] is not None:
                key = f"grid_b{r['b_value']}"
            method_groups[key].append(r)

    print(f"\n  {'Method':<25} {'Mean PPL':>10} {'± Std':>8} {'N':>4} {'Δ Uniform':>12} {'Seeds'}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*4} {'-'*12} {'-'*15}")

    for key in sorted(method_groups.keys()):
        group = method_groups[key]
        ppls = [r["eval_ppl"] for r in group]
        m, s = mean_std(ppls)
        seeds = sorted(set(r["seed"] for r in group if r["seed"]))
        delta = ""
        if uniform_mean and m:
            delta = f"{(m - uniform_mean) / uniform_mean * 100:+.3f}%"
        seed_str = ",".join(str(s) for s in seeds)
        print(f"  {key:<25} {m:>10.4f} {s:>8.4f} {len(ppls):>4} {delta:>12} {seed_str}")

    # ── Grid search: best b per seed ──
    print("\n" + "=" * 90)
    print("  GRID SEARCH: BEST b* PER SEED")
    print("=" * 90)

    grid_rows = [r for r in rows if r["method"] == "grid" and r["eval_ppl"] is not None
                 and r["eval_source"] not in ("trainer_state(fallback)",)]
    seeds_in_grid = sorted(set(r["seed"] for r in grid_rows))
    grid_best_ppls = []

    print(f"\n  {'Seed':>6}  {'Best b':>8}  {'PPL':>10}  {'Δ Uniform':>12}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*12}")
    for seed in seeds_in_grid:
        seed_rows = [r for r in grid_rows if r["seed"] == seed]
        best = min(seed_rows, key=lambda r: r["eval_ppl"])
        grid_best_ppls.append(best["eval_ppl"])
        delta = ""
        if uniform_mean:
            delta = f"{(best['eval_ppl'] - uniform_mean) / uniform_mean * 100:+.3f}%"
        print(f"  {seed:>6}  {best['b_value']:>8.1f}  {best['eval_ppl']:>10.4f}  {delta:>12}")

    if grid_best_ppls:
        gb_m, gb_s = mean_std(grid_best_ppls)
        print(f"\n  Grid-best mean: {gb_m:.4f} ± {gb_s:.4f}")

    # ── Adaptive ratio ──
    if grid_best_ppls and uniform_mean:
        gb_mean = mean_std(grid_best_ppls)[0]
        print("\n" + "=" * 90)
        print("  ADAPTIVE RATIO = (uniform - method) / (uniform - grid_best)")
        print(f"  uniform_mean={uniform_mean:.4f}, grid_best_mean={gb_mean:.4f}")
        print("=" * 90)

        adaptive_methods = ["adaptive", "adaptive_fix", "ess_only", "golden_ratio"]
        for am in adaptive_methods:
            am_rows = [r for r in rows if r["method"] == am and r["eval_ppl"] is not None
                       and r["eval_source"] not in ("trainer_state(fallback)",)]
            if not am_rows:
                continue
            am_ppls = [r["eval_ppl"] for r in am_rows]
            am_m, am_s = mean_std(am_ppls)
            denom = uniform_mean - gb_mean
            if abs(denom) > 1e-6:
                ratio = (uniform_mean - am_m) / denom
            else:
                ratio = float("nan")
            seeds = sorted(set(r["seed"] for r in am_rows))
            print(f"\n  {am:<20} mean={am_m:.4f} ± {am_s:.4f}  ratio={ratio:.3f}  seeds={seeds}")

    # ── Experiments needing eval ──
    need_eval = [r for r in rows if r["status"] == "DONE" and r["eval_source"] in ("none", "trainer_state(fallback)")]
    if need_eval:
        print("\n" + "=" * 90)
        print("  EXPERIMENTS NEEDING STANDALONE EVAL")
        print("=" * 90)
        for r in need_eval:
            fallback = ""
            if r["eval_ppl"] is not None:
                fallback = f" (trainer_state ppl≈{r['eval_ppl']:.2f})"
            print(f"  {r['exp_id']:<40} {r['status']}{fallback}")

    # ── Running experiments ──
    running = [r for r in rows if "RUNNING" in r["status"] or r["status"] == "STARTED"]
    if running:
        print("\n" + "=" * 90)
        print("  RUNNING / IN-PROGRESS EXPERIMENTS")
        print("=" * 90)
        for r in running:
            print(f"  {r['exp_id']:<40} {r['status']}")

    print()


if __name__ == "__main__":
    main()
