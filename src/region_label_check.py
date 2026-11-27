"""What would change if one site's region label followed the boundary register?

Joining the 19 observation points to the OCHA Uganda COD (fetch_boundaries.py)
found a single disagreement: src/config.py registers Masindi in the Northern
region, the COD places Masindi District in the Western region. That label is not
cosmetic. Event exposure in this pipeline is defined by region, not by district
polygons (preprocess.add_labels, ucasi.main, collect_results.main), so a site in
the wrong region moves its district-days between exposure sets.

This module MEASURES that effect and writes it to
results/region_label_sensitivity.json. It changes nothing else: no committed
result, table or figure is re-derived here, because the published numbers were
produced with the labels as they stand. Re-running the whole pipeline under the
corrected label would silently rewrite every figure in the paper, so the
disagreement is disclosed and bounded instead.

Inputs (all persisted):
  data/full_ensemble.csv        operational surface: location, region, date,
                                is_anomaly, label_event
  data/disaster_register.csv    32 events with start/end and affected regions
  data/district_lookup.csv      config name -> COD district and COD region

Run:  python src/region_label_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS


def exposure_labels(df, reg, region_col):
    """Replicate preprocess.add_labels(): 1 where (date, region) is in a window."""
    lab = np.zeros(len(df), dtype=int)
    for _, e in reg.iterrows():
        regs = set(str(e.regions).split(";"))
        mask = ((df.date >= e.start) & (df.date <= e.end)
                & df[region_col].isin(regs)).values
        lab[mask] = 1
    return lab


def window_stats(df, reg, region_col):
    """Per-event exposure: district-days inside the window, and the districts
    that carry a persisted anomaly flag among them."""
    rows = {}
    for _, e in reg.iterrows():
        regs = set(str(e.regions).split(";"))
        m = ((df.date >= e.start) & (df.date <= e.end)
             & df[region_col].isin(regs))
        sub = df[m]
        rows[e.event_id] = {"regions": sorted(regs),
                            "district_days": int(len(sub)),
                            "districts_affected": int(sub.loc[sub.is_anomaly == 1,
                                                              "location"].nunique())}
    return rows


def main(verbose=True):
    ens = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"],
                      usecols=["location", "region", "date", "is_anomaly",
                               "label_event"], low_memory=False)
    reg = pd.read_csv(DATA / "disaster_register.csv", parse_dates=["start", "end"])
    lk = pd.read_csv(DATA / "district_lookup.csv")
    cod = dict(zip(lk.location, lk.adm1_name))
    missing = sorted(set(ens.location) - set(cod))
    if missing:
        raise SystemExit("district_lookup.csv has no COD region for %s" % missing)
    ens["region_cod"] = ens.location.map(cod)

    lab_cfg = exposure_labels(ens, reg, "region")
    lab_cod = exposure_labels(ens, reg, "region_cod")
    agree = float((lab_cfg == ens.label_event.values).mean())
    if agree < 0.99:
        raise SystemExit("replication of the persisted label_event disagrees on "
                         "%.2f%% of rows: this check no longer describes the "
                         "pipeline" % (100 * (1 - agree)))

    flip = np.flatnonzero(lab_cfg != lab_cod)
    lost = int((lab_cfg[flip] == 1).sum())     # positives the COD register drops
    gained = int((lab_cod[flip] == 1).sum())   # positives it adds
    changed = (ens.loc[ens.region != ens.region_cod,
                       ["location", "region", "region_cod"]].drop_duplicates())
    w_cfg = window_stats(ens, reg, "region")
    w_cod = window_stats(ens, reg, "region_cod")
    events = []
    for eid in w_cfg:
        if w_cfg[eid] != w_cod[eid]:
            events.append({"event_id": eid,
                           "regions_config": w_cfg[eid]["regions"],
                           "days_config": w_cfg[eid]["district_days"],
                           "days_cod": w_cod[eid]["district_days"],
                           "affected_config": w_cfg[eid]["districts_affected"],
                           "affected_cod": w_cod[eid]["districts_affected"]})

    out = {
        "rows_total": int(len(ens)),
        "labelled_positive_config": int(lab_cfg.sum()),
        "labelled_positive_cod": int(lab_cod.sum()),
        "rows_whose_label_would_change": int(len(flip)),
        "positives_lost": lost,
        "positives_gained": gained,
        "share_of_current_positives_lost": round(lost / max(lab_cfg.sum(), 1), 5),
        "max_shift_in_districts_affected":
            max([abs(e["affected_cod"] - e["affected_config"])
                 for e in events], default=0),
        "agreement_with_persisted_label_event": round(agree, 6),
        "sites_whose_region_label_differs":
            changed.rename(columns={"region": "config_region",
                                    "region_cod": "cod_region"}).to_dict("records"),
        "events_with_changed_exposure": events,
        "note": ("Measured on persisted artefacts only. No committed result was "
                 "re-derived: the published numbers were produced with the "
                 "region labels as registered in src/config.py."),
    }
    p = RESULTS / "region_label_sensitivity.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if verbose:
        print("wrote %s" % p.relative_to(C.ROOT))
        print("  rows %d, positives %d -> %d under the COD region"
              % (out["rows_total"], out["labelled_positive_config"],
                 out["labelled_positive_cod"]))
        print("  labels that would flip: %d (%d dropped, %d added; %.2f%% of "
              "current positives dropped)"
              % (out["rows_whose_label_would_change"], lost, gained,
                 100 * out["share_of_current_positives_lost"]))
        print("  largest shift in districts_affected: %d"
              % out["max_shift_in_districts_affected"])
        print("  events with any exposure change: %d of %d"
              % (len(events), len(w_cfg)))
        for e in events:
            print("    %s %s days %d->%d affected %d->%d"
                  % (e["event_id"], "+".join(e["regions_config"]), e["days_config"],
                     e["days_cod"], e["affected_config"], e["affected_cod"]))
    return out


if __name__ == "__main__":
    main()
