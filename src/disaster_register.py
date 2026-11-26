"""Curated Uganda climate-disaster ground-truth register (2008-2023).

Events are compiled from published humanitarian/meteorological records
(Uganda National Meteorological Authority - UNMA, OCHA/ReliefWeb, EM-DAT,
FAO/GIEWS, PreventionWeb, and peer-reviewed Uganda climate-disaster reports).
Each entry is a documented hydro-meteorological event with an onset window
and the administrative region(s) affected. This register provides the
supervision labels used to VALIDATE unsupervised anomaly detection (RQ5),
addressing the key methodological gap in the baseline study (which reported
precision/recall/AUC without any ground truth).

Columns: event_id, start, end, type, regions, magnitude_note, source
type in {Flood, Drought, Landslide, Storm}
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# (start, end, type, [regions], magnitude_note, primary source)
EVENTS = [
    ("2010-03-01", "2010-05-31", "Flood",     ["Eastern", "Central"], "Long-rains flooding, multiple districts", "UNMA/OCHA 2010"),
    ("2010-11-29", "2010-12-05", "Landslide", ["Eastern"], "Bududa landslides, >200 deaths, ~4,900 displaced", "OCHA/ReliefWeb 2010"),
    ("2010-12-01", "2011-06-30", "Drought",   ["Northern"], "Severe 2010-11 drought, Karamoja food crisis", "FAO/GIEWS 2011"),
    ("2011-04-01", "2011-05-31", "Flood",     ["Western", "Central"], "Bundibugyo/Kibimba floods", "OCHA 2011"),
    ("2011-05-08", "2011-05-13", "Landslide", ["Western"], "Bundibugyo & Kisoro mudslides, ~120 deaths", "ReliefWeb 2011"),
    ("2011-08-01", "2011-08-31", "Flood",     ["Eastern", "Northern"], "Soroti/Kumi flood displacement", "OCHA 2011"),
    ("2012-10-01", "2012-10-31", "Flood",     ["Eastern"], "Teso (Katakwi/Kumi/Pallisa) floods", "UNMA 2012"),
    ("2013-11-01", "2013-11-30", "Landslide", ["Eastern"], "Bududa/Sironko landslides, ~55 deaths", "OCHA/ReliefWeb 2013"),
    ("2014-03-01", "2014-05-31", "Flood",     ["Eastern", "Central"], "Bugiri/Busia/Iganga long-rains floods", "UNMA 2014"),
    ("2014-09-01", "2014-10-31", "Flood",     ["Western", "Central"], "Sept-Oct short-rains flooding", "OCHA 2014"),
    ("2015-04-01", "2015-05-31", "Flood",     ["Central", "Western"], "Kampala & Western floods, deaths", "UNMA 2015"),
    ("2015-11-25", "2015-12-05", "Landslide", ["Central"], "Butambala/Bukino mudslides, El Nino rains", "ReliefWeb 2015"),
    ("2016-01-01", "2016-08-31", "Drought",   ["Northern"], "2016 El Nino drought, Karamoja/NE", "FAO/FEWS NET 2016"),
    ("2016-04-01", "2016-05-31", "Flood",     ["Western"], "Bushenyi/Kisoro/Bundibugyo floods", "OCHA 2016"),
    ("2018-01-01", "2018-06-30", "Drought",   ["Northern", "Eastern"], "2017-18 dry spells -> food crisis Karamoja/Teso/Lango", "FEWS NET 2018"),
    ("2018-03-01", "2018-05-31", "Flood",     ["Eastern", "Western"], "Bududa/Mbale & Buliisa floods/landslides", "UNMA 2018"),
    ("2019-05-01", "2019-05-15", "Landslide", ["Central", "Western"], "Kalisunga (Mukono) & Marinda (Rakai) mudslides, ~40+ deaths", "ReliefWeb/ODPPP 2019"),
    ("2019-11-01", "2019-12-31", "Flood",     ["Eastern", "Central"], "Busia/Iganga/Kiboga Nov-Dec 2019 floods", "UNMA/OCHA 2019"),
    ("2019-11-01", "2019-12-31", "Landslide", ["Western"], "Bundibugyo/Kabale Dec 2019 landslides", "OCHA 2019"),
    ("2020-03-01", "2020-03-31", "Landslide", ["Western"], "Buliisa/Hoima landslides & floods", "ReliefWeb 2020"),
    ("2020-05-01", "2020-05-31", "Flood",     ["Central", "Western"], "May 2020 Kampala & Western flooding", "UNMA 2020"),
    ("2020-10-01", "2021-03-31", "Drought",   ["Northern"], "Below-normal rains, water shortage, NE Uganda", "UNMA/FEWS NET 2020"),
    ("2021-09-01", "2021-09-30", "Flood",     ["Eastern"], "Busia/Bududa Sept 2021 floods", "UNMA 2021"),
    ("2021-10-01", "2021-10-31", "Flood",     ["Central", "Eastern"], "Kampala/Mukono/Jinja Oct 2021 floods", "OCHA 2021"),
    ("2021-11-01", "2021-11-30", "Landslide", ["Western"], "Bundibugyo/Kanungu/Kisoro Nov 2021 landslides, ~30 deaths", "ReliefWeb 2021"),
    ("2022-05-01", "2022-05-31", "Landslide", ["Western"], "Bundibugyo/Kisoro/Kabale May 2022 landslides", "OCHA 2022"),
    ("2022-07-01", "2022-07-31", "Landslide", ["Central"], "Mubende/Mityana land subsidence, ~5,000 displaced", "ReliefWeb/ODPPP 2022"),
    ("2022-09-01", "2022-09-30", "Flood",     ["Western"], "Kamwenge/Kyenjojo/Fort Portal Sept 2022 floods", "UNMA 2022"),
    ("2022-10-15", "2022-10-22", "Landslide", ["Western"], "Buliisa (Moyok/Kagadi) Oct 2022 landslide, deaths", "ReliefWeb 2022"),
    ("2023-03-01", "2023-05-31", "Flood",     ["Eastern"], "Bududa/Elgon & Teso long-rains floods", "UNMA 2023"),
    ("2023-09-01", "2023-09-30", "Flood",     ["Northern"], "Karamoja Sept 2023 floods", "OCHA 2023"),
    ("2023-10-10", "2023-10-20", "Landslide", ["Western"], "Kasese/Kitagwenda Oct 2023 floods & landslides, deaths", "ReliefWeb 2023"),
]

TYPES = {"Flood", "Drought", "Landslide", "Storm"}


def build():
    rows = []
    for i, (s, e, t, regs, mag, src) in enumerate(EVENTS, 1):
        rows.append({
            "event_id": f"UG-{i:03d}",
            "start": pd.to_datetime(s),
            "end": pd.to_datetime(e),
            "type": t,
            "regions": ";".join(regs),
            "magnitude_note": mag,
            "source": src,
        })
    df = pd.DataFrame(rows)
    df = df[(df["start"] >= pd.Timestamp(C.START_DATE)) & (df["start"] <= pd.Timestamp(C.END_DATE))]
    df.to_csv(C.DATA / "disaster_register.csv", index=False)
    return df


if __name__ == "__main__":
    df = build()
    print(f"events={len(df)}  types={dict(df.type.value_counts())}")
    print(f"regions touched: {sorted({r for rs in df.regions for r in rs.split(';')})}")
    print(df.head(10).to_string(index=False))
