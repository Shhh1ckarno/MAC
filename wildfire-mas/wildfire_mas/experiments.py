from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import MECHANISMS, SimConfig
from .sim import Simulation

MECH_RU = {
    "central_greedy": "Диспетчер жадный",
    "central_hungarian": "Диспетчер венгерский",
    "cnp": "Контрактная сеть",
    "auction_first": "Аукцион, 1-я цена",
    "auction_vickrey": "Аукцион Викри",
    "cnp_drone": "Торги от дронов",
}
SCEN_RU = {"default": "Базовый", "windy": "Сильный ветер", "outbreak": "Массовые возгорания"}
COLORS = {
    "central_greedy": "#2a78d6",
    "central_hungarian": "#eb6834",
    "cnp": "#1baf7a",
    "auction_first": "#eda100",
    "auction_vickrey": "#e87ba4",
    "cnp_drone": "#008300",
}
FIELDS = ["series", "scenario", "mechanism", "seed", "n_crews", "p_loss", "J_burn", "J_burn_base", "Save", "J_set",
          "T_ext_mean", "detect_latency_mean", "response_time_mean", "M_delivered", "M_sent", "bytes",
          "coord_time_ms", "requests", "refusals", "reassignments", "gini_busy", "refill_wait_mean", "wall_time_s"]


def _run(job: dict) -> dict:
    cfg = SimConfig(**job["cfg"])
    t0 = time.perf_counter()
    r = Simulation(cfg).run()
    r["wall_time_s"] = round(time.perf_counter() - t0, 3)
    r["job"] = job
    return r


def _jobs(seeds: list[int]) -> list[dict]:
    jobs = []

    def add(series, scenario, mech, seed, n_crews=5, p_loss=0.0):
        cfg = {"seed": seed, "scenario": scenario, "n_crews": n_crews, "p_loss_cmd": p_loss, "p_loss_radio": p_loss}
        if mech == "none":
            cfg["agents_enabled"] = False
        else:
            cfg["mechanism"] = mech
        jobs.append({"series": series, "scenario": scenario, "mechanism": mech, "seed": seed,
                     "n_crews": n_crews, "p_loss": p_loss, "cfg": cfg})

    for sc in SCEN_RU:
        for s in seeds:
            add("main", sc, "none", s)
            for m in MECHANISMS:
                add("main", sc, m, s)
    small = seeds[:max(10, len(seeds) * 2 // 3)]
    for n in (3, 8):
        for s in small:
            for m in MECHANISMS:
                add("scale", "default", m, s, n_crews=n)
    for p in (0.1, 0.3):
        for s in small:
            for m in MECHANISMS:
                add("loss", "default", m, s, p_loss=p)
    return jobs


def run_lab2(out: Path, n_seeds: int = 30, workers: int | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, n_seeds + 1))
    jobs = _jobs(seeds)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_run, jobs, chunksize=4), 1):
            j = r["job"]
            rows.append({**{k: j[k] for k in ("series", "scenario", "mechanism", "seed", "n_crews", "p_loss")},
                         **{k: r.get(k) for k in FIELDS if k in r}})
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} прогонов, {time.perf_counter() - t0:.0f} с", flush=True)
    base = {(r["scenario"], r["seed"]): r["J_burn"] for r in rows if r["mechanism"] == "none"}
    for r in rows:
        b = base.get(("default" if r["series"] != "main" else r["scenario"], r["seed"]))
        r["J_burn_base"] = b
        r["Save"] = round(1 - r["J_burn"] / b, 4) if b else None
    path = out / "lab2_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print(f"Сохранено {len(rows)} прогонов в {path} за {time.perf_counter() - t0:.0f} с")
    return path


def load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k in ("series", "scenario", "mechanism"):
                    continue
                r[k] = float(v) if v not in ("", "None") else math.nan
            rows.append(r)
    return rows


def mean_ci(x) -> tuple[float, float]:
    x = np.asarray([v for v in x if not math.isnan(v)], dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else math.nan, 0.0)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def paired_diff(a, b, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def summarize(rows: list[dict]) -> dict:
    out = {"main": {}, "paired": {}, "scale": {}, "loss": {}}
    for sc in SCEN_RU:
        base_rows = _sel(rows, series="main", scenario=sc, mechanism="none")
        out["main"][sc] = {"base": mean_ci([r["J_burn"] for r in base_rows])}
        ref = {r["seed"]: r for r in _sel(rows, series="main", scenario=sc, mechanism="central_greedy")}
        for m in MECHANISMS:
            rs = _sel(rows, series="main", scenario=sc, mechanism=m)
            out["main"][sc][m] = {k: mean_ci([r[k] for r in rs]) for k in
                                  ("J_burn", "Save", "J_set", "T_ext_mean", "response_time_mean", "M_delivered",
                                   "coord_time_ms", "gini_busy", "reassignments")}
            out["main"][sc][m]["lost_any"] = sum(1 for r in rs if r["J_set"] > 0)
            out["main"][sc][m]["n"] = len(rs)
            if m != "central_greedy":
                pairs = [(r["J_burn"], ref[r["seed"]]["J_burn"]) for r in rs if r["seed"] in ref]
                out["paired"].setdefault(sc, {})[m] = paired_diff([p[0] for p in pairs], [p[1] for p in pairs])
    for series, key, values in (("scale", "n_crews", (3, 5, 8)), ("loss", "p_loss", (0.0, 0.1, 0.3))):
        seeds_small = sorted({r["seed"] for r in rows if r["series"] == series})
        for m in MECHANISMS:
            for v in values:
                if (series == "scale" and v == 5) or (series == "loss" and v == 0.0):
                    rs = [r for r in _sel(rows, series="main", scenario="default", mechanism=m) if r["seed"] in seeds_small]
                else:
                    rs = [r for r in _sel(rows, series=series, mechanism=m) if r[key] == v]
                out[series].setdefault(m, {})[v] = {k: mean_ci([r[k] for r in rs]) for k in ("Save", "M_delivered", "J_burn")}
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ad")
    ax.tick_params(colors="#3a3a37", labelsize=10)
    ax.grid(axis="x", color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)


def _fmt(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def plots(summary: dict, out: Path, merge_auctions: bool = True) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "text.color": "#1d1d1b",
                         "axes.labelcolor": "#3a3a37"})
    files = []
    mechs = [m for m in MECHANISMS if not (merge_auctions and m == "auction_vickrey")]
    names = dict(MECH_RU)
    if merge_auctions:
        names["auction_first"] = "Аукцион (обе цены)"

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6), sharey=True)
    for ax, sc in zip(axes, SCEN_RU):
        for i, m in enumerate(mechs):
            mu, ci = summary["main"][sc][m]["Save"]
            ax.plot([mu - ci, mu + ci], [i, i], color="#3a3a37", linewidth=1.5, solid_capstyle="round")
            ax.scatter([mu], [i], s=70, color=COLORS[m], edgecolor="white", linewidth=2, zorder=3)
            ax.text(min(mu + ci, 1.0) + 0.02, i, _fmt(mu), va="center", fontsize=9.5, color="#3a3a37")
        ax.set_title(SCEN_RU[sc], fontsize=12, loc="left", color="#1d1d1b")
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1.0], [_fmt(v, 1) for v in (0, 0.5, 1.0)])
        ax.set_xlabel("Save")
        _style(ax)
    axes[0].set_yticks(range(len(mechs)), [names[m] for m in mechs])
    axes[0].invert_yaxis()
    fig.tight_layout()
    files.append(out / "save_by_scenario.png")
    fig.savefig(files[-1], dpi=200, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for m in mechs:
        x, xci = summary["main"]["default"][m]["M_delivered"]
        y, yci = summary["main"]["default"][m]["Save"]
        ax.errorbar([x], [y], xerr=[xci], yerr=[yci], fmt="none", ecolor="#8a8980", elinewidth=1, capsize=0)
        ax.scatter([x], [y], s=90, color=COLORS[m], edgecolor="white", linewidth=2, zorder=3)
        off = {"central_greedy": (8, 8), "central_hungarian": (8, -16)}.get(m, (8, 6))
        ax.annotate(names[m], (x, y), xytext=off, textcoords="offset points", fontsize=9.5, color="#1d1d1b")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt(v)))
    ax.set_xlabel("Сообщений за прогон (доставлено)")
    ax.set_ylabel("Доля спасенной ценности (Save)")
    _style(ax)
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
    fig.tight_layout()
    files.append(out / "messages_vs_save.png")
    fig.savefig(files[-1], dpi=200, facecolor="white")
    plt.close(fig)

    for series, key_ru, values, name in (("scale", "Число расчетов", (3, 5, 8), "scale_save.png"),
                                         ("loss", "Доля потерянных сообщений", (0.0, 0.1, 0.3), "loss_save.png")):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for m in mechs:
            ys = [summary[series][m][v]["Save"][0] for v in values]
            ax.plot(values, ys, color=COLORS[m], linewidth=2, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=names[m])
        ax.set_xticks(values, [str(v) if series == "scale" else f"{int(v * 100)} %" for v in values])
        ax.set_xlabel(key_ru)
        ax.set_ylabel("Доля спасенной ценности (Save)")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [_fmt(v, 1) for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)])
        _style(ax)
        ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
        ax.legend(frameon=False, fontsize=9.5, loc="lower right" if series == "scale" else "lower left")
        fig.tight_layout()
        files.append(out / name)
        fig.savefig(files[-1], dpi=200, facecolor="white")
        plt.close(fig)
    return files


def analyze(csv_path: Path, out: Path) -> dict:
    rows = load(csv_path)
    s = summarize(rows)
    first = {(r["scenario"], r["seed"], r["series"], r["n_crews"], r["p_loss"]): r["J_burn"]
             for r in rows if r["mechanism"] == "auction_first"}
    vick = {(r["scenario"], r["seed"], r["series"], r["n_crews"], r["p_loss"]): r["J_burn"]
            for r in rows if r["mechanism"] == "auction_vickrey"}
    s["auctions_identical"] = first == vick
    (out / "lab2_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    files = plots(s, out, merge_auctions=s["auctions_identical"])
    print("Графики:", ", ".join(str(f) for f in files))
    return s

LLM_FIELDS = ["scenario", "seed", "J_burn", "J_set", "T_ext_mean", "response_time_mean", "M_delivered",
              "coord_time_ms", "requests", "refusals", "calls", "ok", "invalid", "errors", "fallbacks", "violations",
              "mean_latency", "wall_time_s"]


def run_llm_series(out: Path, n_seeds: int = 15, scenarios=tuple(SCEN_RU)) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "lab2_llm.csv"
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            done = {(r["scenario"], int(r["seed"])) for r in csv.DictReader(f)}
    new = not path.exists()
    total = len(scenarios) * n_seeds
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LLM_FIELDS)
        if new:
            w.writeheader()
        for sc in scenarios:
            for seed in range(1, n_seeds + 1):
                if (sc, seed) in done:
                    continue
                t0 = time.perf_counter()
                r = Simulation(SimConfig(seed=seed, scenario=sc, hq_mode="llm")).run()
                co = r.get("coordinator", {})
                row = {**{k: r.get(k) for k in LLM_FIELDS}, **{k: co.get(k) for k in LLM_FIELDS if k in co},
                       "scenario": sc, "seed": seed, "wall_time_s": round(time.perf_counter() - t0, 1)}
                w.writerow(row)
                f.flush()
                done.add((sc, seed))
                print(f"{len(done)}/{total} {sc} seed={seed} J_burn={r['J_burn']} вызовов={co.get('calls')} "
                      f"ошибок={co.get('errors')} откатов={co.get('fallbacks')} {row['wall_time_s']} с", flush=True)
    return path


def analyze_llm(out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    llm = load(out / "lab2_llm.csv")
    rules = load(out / "lab2_runs.csv")
    seeds = sorted({int(r["seed"]) for r in llm})
    res = {}
    compare = ["central_greedy", "central_hungarian", "auction_first", "llm"]
    names = {**MECH_RU, "auction_first": "Аукцион (обе цены)", "llm": "LLM-штаб (Qwen 3.5 9B)"}
    colors = {**COLORS, "llm": "#7b5cd6"}
    for sc in SCEN_RU:
        lr = {int(r["seed"]): r for r in llm if r["scenario"] == sc}
        if not lr:
            continue
        base = {int(r["seed"]): r["J_burn"] for r in rules if r["series"] == "main" and r["scenario"] == sc
                and r["mechanism"] == "none"}
        res[sc] = {}
        for m in compare:
            if m == "llm":
                rs = [dict(lr[s], Save=1 - lr[s]["J_burn"] / base[s]) for s in sorted(lr)]
            else:
                rs = [r for r in rules if r["series"] == "main" and r["scenario"] == sc and r["mechanism"] == m
                      and int(r["seed"]) in lr]
            res[sc][m] = {k: mean_ci([r[k] for r in rs]) for k in ("J_burn", "Save", "M_delivered", "response_time_mean",
                                                                   "T_ext_mean", "coord_time_ms")}
            res[sc][m]["lost_any"] = sum(1 for r in rs if r["J_set"] > 0)
            res[sc][m]["n"] = len(rs)
        greedy = {int(r["seed"]): r["J_burn"] for r in rules if r["series"] == "main" and r["scenario"] == sc
                  and r["mechanism"] == "central_greedy"}
        ss = sorted(lr)
        res[sc]["paired_llm_minus_greedy"] = paired_diff([lr[s]["J_burn"] for s in ss], [greedy[s] for s in ss])
        res[sc]["llm_stats"] = {k: float(np.nansum([lr[s][k] for s in ss])) for k in
                                ("calls", "ok", "invalid", "errors", "fallbacks", "violations")}
        res[sc]["llm_stats"]["mean_latency"] = float(np.nanmean([lr[s]["mean_latency"] for s in ss]))
        res[sc]["llm_stats"]["wall_time_s"] = float(np.nanmean([lr[s]["wall_time_s"] for s in ss]))
    (out / "lab2_llm_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                               encoding="utf-8")
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    scs = list(res)
    fig, axes = plt.subplots(1, len(scs), figsize=(3.2 * len(scs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scs):
        for i, m in enumerate(compare):
            mu, ci = res[sc][m]["Save"]
            ax.plot([mu - ci, min(mu + ci, 1.0)], [i, i], color="#3a3a37", linewidth=1.5)
            ax.scatter([mu], [i], s=70, color=colors[m], edgecolor="white", linewidth=2, zorder=3)
            ax.text(min(mu + ci, 1.0) + 0.02, i, _fmt(mu), va="center", fontsize=9.5, color="#3a3a37")
        ax.set_title(SCEN_RU[sc], fontsize=12, loc="left")
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1.0], [_fmt(v, 1) for v in (0, 0.5, 1.0)])
        ax.set_xlabel("Save")
        _style(ax)
    axes[0].set_yticks(range(len(compare)), [names[m] for m in compare])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "llm_vs_rules.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float)[:3000])
    return res
