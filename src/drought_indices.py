"""Derived drought indices: SPI (precipitation) and SPEI (precip - PET).

Fills the Tier-5 gap in the variable set: raw anomalies answer "is today
unusual?", but SPI/SPEI answer "is there an agricultural/meteorological
DROUGHT, and how severe is it?" -- McKee et al. (1993) and Vicente-Serrano
et al. (2010).

Design choices (all justified by what the data can honestly support):
  * SPI is computed on the LONG Meteostat precipitation record (~1926-2026,
    ~100 years) so the per-calendar-month gamma fits have enough samples to be
    statistically defensible. ERA5 is only 16 years -- too thin for SPI-12.
  * SPEI needs a PET; only ERA5 carries FAO-56 ET0, so SPEI is computed on the
    16-year ERA5 window and explicitly labelled as a short-record index. Its
    water balance can be negative, so we fit a Pearson III (generalised
    gamma) per calendar month, a recognised alternative for the difference
    series, rather than the zero-inflated gamma used for SPI.
  * Zero-precipitation months are handled with McKee's reduced-probability
    adjustment Pe = q + (1-q)G(x) where q is the within-month zero fraction.

Outputs:
  results/tables/spi_meteostat.csv   station x month x {SPI1,SPI3,SPI6,SPI12}
  results/tables/spei_era5.csv       district x month x {SPI3,SPI6,SPEI3,SPEI6}
  results/drought_summary.json       validation stats + register-drought check
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS
from scipy import stats


def _spi_from_accum(accum, month, periods):
    """McKee-style SPI per accumulation period using per-calendar-month gamma.

    accum : Series of the k-month accumulated value, index = DatetimeIndex
    month : Series of calendar months aligned to accum.index
    returns dict period -> Series (SPI values, NaN where fit infeasible)
    """
    out = {}
    for k, s in periods.items():
        y = s.dropna()
        if len(y) < 24:
            out[k] = pd.Series(np.nan, index=accum.index)
            continue
        spi = pd.Series(np.nan, index=y.index, dtype=float)
        for m in range(1, 13):
            sel = y[month[y.index] == m]
            v = sel.values
            if len(v) < 12:
                continue
            q = float((v <= 0).mean())                # zero fraction
            pos = v[v > 0]
            if len(pos) < 8:                          # too few positives to fit
                continue
            try:
                shape, loc, scale = stats.gamma.fit(pos, floc=0)
            except Exception:
                continue
            if not np.isfinite([shape, scale]).all() or shape <= 0 or scale <= 0:
                continue
            G = stats.gamma.cdf(sel.values, shape, loc, scale)
            Pe = q + (1.0 - q) * G                    # reduced probability
            Pe = np.clip(Pe, 1e-6, 1 - 1e-6)
            spi.loc[sel.index] = stats.norm.ppf(Pe)
        out[k] = spi.reindex(accum.index)
    return out


def _acc(series, k):
    """k-month rolling accumulation; NaN until k complete months exist."""
    return series.rolling(k, min_periods=k).sum()


def meteostat_spi(met, min_days=22):
    """Century-scale SPI per station on monthly precipitation.

    Historical Meteostat daily coverage is sparse; a monthly sum built from a
    handful of observed days badly under-counts true precipitation, which
    corrupts the gamma fit. We therefore keep a month only if >= min_days were
    observed, so the SPI substrate is the well-covered part of the record.
    """
    frames = []
    for sid, g in met.groupby("station"):
        g = g.copy()
        g["date"] = pd.to_datetime(g["date"])
        g["prcp"] = pd.to_numeric(g["prcp"], errors="coerce")
        g = g.dropna(subset=["prcp"]).sort_values("date")
        monthly = g.set_index("date")["prcp"].resample("MS")
        ndays = monthly.count()
        m = monthly.sum().where(ndays >= min_days)   # NaN incomplete months
        m = m.dropna()
        if len(m) < 60:
            continue
        mon = pd.Series(m.index.month, index=m.index)
        periods = {k: _acc(m, k) for k in (1, 3, 6, 12)}
        spi = _spi_from_accum(periods[1], mon, periods)
        df = pd.DataFrame({"station": str(sid), "name": g["name"].iloc[0],
                           "month": m.index, "prcp_mm": m.values})
        for k, s in spi.items():
            df[f"SPI{k}"] = np.round(s.values, 3)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["station", "name", "month", "prcp_mm",
                                     "SPI1", "SPI3", "SPI6", "SPI12"])
    return pd.concat(frames, ignore_index=True)


def _spei_from_balance(bal, month, periods):
    """SPEI via per-calendar-month Pearson III fit on the P-PET balance."""
    out = {}
    for k, s in periods.items():
        y = s.dropna()
        if len(y) < 24:
            out[k] = pd.Series(np.nan, index=bal.index)
            continue
        val = pd.Series(np.nan, index=y.index, dtype=float)
        for m in range(1, 13):
            sel = y[month[y.index] == m]
            v = sel.values
            if len(v) < 12 or np.std(v) <= 0:
                continue
            try:
                skew, loc, scale = stats.pearson3.fit(v)
            except Exception:
                continue
            if not np.isfinite([skew, loc, scale]).all() or scale <= 0:
                continue
            Pe = stats.pearson3.cdf(sel.values, skew, loc, scale)
            Pe = np.clip(Pe, 1e-6, 1 - 1e-6)
            val.loc[sel.index] = stats.norm.ppf(Pe)
        out[k] = val.reindex(bal.index)
    return out


def era5_spi_spei(era):
    """SPI and SPEI on ERA5 (16 yr) for the register/impact window."""
    frames = []
    for loc, g in era.groupby("location"):
        g = g.copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.dropna(subset=["precipitation_sum"]).sort_values("date")
        pm = g.set_index("date")["precipitation_sum"].resample("MS").sum()
        pet = pd.to_numeric(g.set_index("date")["et0_fao_evapotranspiration"],
                            errors="coerce").resample("MS").sum()
        mon = pd.Series(pm.index.month, index=pm.index)
        spi_p = {k: _acc(pm, k) for k in (3, 6)}
        spi = _spi_from_accum(spi_p[3], mon, spi_p)
        bal = pm - pet
        spei_p = {k: _acc(bal, k) for k in (3, 6)}
        spei = _spei_from_balance(bal, mon, spei_p)
        df = pd.DataFrame({"location": loc, "region": g["region"].iloc[0],
                           "month": pm.index, "prcp_mm": np.round(pm.values, 2),
                           "pet_mm": np.round(pet.values, 2)})
        for k, s in spi.items():
            df[f"SPI{k}"] = np.round(s.values, 3)
        for k, s in spei.items():
            df[f"SPEI{k}"] = np.round(s.values, 3)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def decade_drought(spi, col="SPI12"):
    """Fraction of months in SPI< -1.0 / -1.5 / -2.0 by decade (century record)."""
    s = spi.dropna(subset=[col]).copy()
    s["decade"] = (pd.to_datetime(s["month"]).dt.year // 10) * 10
    rows = []
    for d, g in s.groupby("decade"):
        n = len(g)
        if n < 24:
            continue
        rows.append(dict(decade=int(d), n_months=int(n),
                         drought_freq=round(float((g[col] < -1.0).mean()), 3),
                         severe_freq=round(float((g[col] < -1.5).mean()), 3),
                         extreme_freq=round(float((g[col] < -2.0).mean()), 3)))
    return pd.DataFrame(rows).sort_values("decade")


def validate(series, label):
    s = np.asarray(pd.Series(series).dropna(), float)
    return {label: {"n": int(len(s)), "mean": round(float(np.mean(s)), 3) if len(s) else None,
                    "sd": round(float(np.std(s)), 3) if len(s) else None,
                    "min": round(float(np.min(s)), 2) if len(s) else None,
                    "max": round(float(np.max(s)), 2) if len(s) else None}}


def main():
    met = pd.read_csv(DATA / "meteostat_daily.csv")
    spi = meteostat_spi(met)
    spi.to_csv(RESULTS / "tables/spi_meteostat.csv", index=False)

    era = pd.read_csv(DATA / "era5_daily.csv")
    es = era5_spi_spei(era)
    es.to_csv(RESULTS / "tables/spei_era5.csv", index=False)

    # validation: indices should be ~N(0,1)
    stats_check = {}
    for c in ["SPI1", "SPI3", "SPI6", "SPI12"]:
        stats_check.update(validate(spi[c], f"meteostat_{c}"))
    for c in ["SPI3", "SPI6", "SPEI3", "SPEI6"]:
        stats_check.update(validate(es[c], f"era5_{c}"))

    # drought-frequency trend on the century record
    dd = decade_drought(spi, "SPI6")
    dd.to_csv(RESULTS / "tables/spi_decade_drought.csv", index=False)

    # ---- register-drought validation: SPI/SPEI during documented events ----
    # The register is REGION-scoped (e.g. 'Northern', 'Eastern;Central'), so we
    # ask: across the ERA5 districts in the affected region, over the event
    # window, was the derived index actually in drought (<= -1.0)?
    reg = pd.read_csv(DATA / "disaster_register.csv")
    dr = reg[reg["type"].str.lower().str.contains("drought", na=False)]
    es_m = es.copy()
    es_m["month"] = pd.to_datetime(es_m["month"])
    event_check = []
    for _, e in dr.iterrows():
        s = pd.to_datetime(str(e["start"]), errors="coerce")
        en = pd.to_datetime(str(e["end"]), errors="coerce")
        if pd.isna(s):
            continue
        en = en if pd.notna(en) else s
        regs = [r.strip() for r in str(e["regions"]).split(";") if r.strip()]
        win = es_m[(es_m.month >= s.to_period("M").to_timestamp()) &
                   (es_m.month <= en.to_period("M").to_timestamp()) &
                   (es_m.region.isin(regs))]
        if not len(win):
            continue
        sp3 = pd.to_numeric(win["SPEI3"], errors="coerce")
        sp6 = pd.to_numeric(win["SPEI6"], errors="coerce")
        si6 = pd.to_numeric(win["SPI6"], errors="coerce")
        event_check.append(dict(event=e["event_id"], regions=";".join(regs),
                                window=f"{s.date()}..{en.date()}", n_district_months=int(len(win)),
                                min_SPEI3=round(float(np.nanmin(sp3)), 2) if sp3.notna().any() else None,
                                min_SPEI6=round(float(np.nanmin(sp6)), 2) if sp6.notna().any() else None,
                                min_SPI6=round(float(np.nanmin(si6)), 2) if si6.notna().any() else None))
    wet = [ev for ev in event_check if (ev.get("min_SPEI3") is not None and ev["min_SPEI3"] <= -1.0)
           or (ev.get("min_SPI6") is not None and ev["min_SPI6"] <= -1.0)]
    summary = {
        "spi_stations": int(spi.station.nunique()),
        "spi_months": int(len(spi)),
        "spi_record_start": str(pd.to_datetime(spi.month).min().date()),
        "spi_record_end": str(pd.to_datetime(spi.month).max().date()),
        "spei_locations": int(es.location.nunique()),
        "spei_months": int(len(es)),
        "index_validation": stats_check,
        "worst_century_droughts": spi.dropna(subset=["SPI12"]).nsmallest(8, "SPI12")
            [["name", "month", "SPI12"]].assign(month=lambda d: pd.to_datetime(d.month).dt.strftime("%Y-%m")).to_dict("records"),
        "decade_drought_trend": dd.to_dict("records"),
        "register_droughts_checked": len(event_check),
        "register_droughts_with_spei_le_minus1": len(wet),
        "register_drought_spei": event_check,
    }
    (RESULTS / "drought_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("SPI stations:", summary["spi_stations"], " months:", summary["spi_months"],
          f"({summary['spi_record_start']}..{summary['spi_record_end']})")
    print("SPEI locations:", summary["spei_locations"], " months:", summary["spei_months"])
    print("\nindex validation (want mean~0, sd~1):")
    for k, v in stats_check.items():
        print(f"  {k:20s} n={v['n']:5d} mean={v['mean']} sd={v['sd']} min={v['min']} max={v['max']}")
    print("\ndecade drought frequency (SPI6 < -1):")
    print(dd.to_string(index=False))
    print("\nworst century droughts (SPI12):")
    for r in summary["worst_century_droughts"]:
        print(f"  {r['name']:18s} {r['month']}  SPI12={r['SPI12']}")
    print("\nregister-drought SPEI/SPI check (region-window minima):")
    for ev in event_check:
        print(f"  {ev['event']:8s} {ev['regions']:16s} {ev['window']}  "
              f"minSPEI3={ev['min_SPEI3']} minSPEI6={ev['min_SPEI6']} minSPI6={ev['min_SPI6']}")
    print(f"  -> {len(wet)}/{len(event_check)} documented droughts show SPEI3 or SPI6 <= -1")


if __name__ == "__main__":
    main()
