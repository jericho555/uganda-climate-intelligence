"""Figure for the World Bank portal cross-check (src/wb_climate_check.py).

Two panels, both drawn from the artefacts the check writes -- nothing is typed in
by hand:

  panel A  annual SPEI: the Bank's national index (EN.CLC.SPEI.XD, 1990-2023)
           against the mean and the minimum of the SPEI6 we derive from ERA5, with
           every register-drought year shaded. The point of the panel is the
           DISAGREEMENT: an independent national annual index can be wet in the
           very years Uganda's disaster register records droughts.
  panel B  the level check: district-mean annual precipitation from ERA5 against
           the Bank's long-term national normal, with the wettest and driest sites
           of our own 16-site panel drawn as reference lines so the reader can see
           how unrepresentative a 16-centroid sample is of a whole country.

Output -> figures/ and manuscript/figures/ (config.save_figure), as fig_wb_check.
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})           # dpi/bbox owned by config.save_figure

CHECK = C.RESULTS / "wb_climate_check.json"
SPEI_TAB = C.RESULTS / "tables" / "wb_spei_compare.csv"
PRECIP_TAB = C.RESULTS / "tables" / "wb_precip_compare.csv"


def main():
    for p in (CHECK, SPEI_TAB, PRECIP_TAB):
        if not p.exists():
            raise SystemExit(f"{p.name} missing -- run src/wb_climate_check.py first")
    js = json.loads(CHECK.read_text(encoding="utf-8"))
    sp = pd.read_csv(SPEI_TAB)
    pr = pd.read_csv(PRECIP_TAB)
    a = js["agreement_vs_ours_spei6_mean"]
    pc = js["precip_normal_check"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    dr = sp[sp.register_drought_year == 1].year
    for y in dr:
        ax.axvspan(y - 0.45, y + 0.45, color="#e8b4b8", alpha=0.45, lw=0)
    ax.axvspan(C.EVAL_TRAIN_YEARS[0] - 0.5, C.EVAL_TEST_YEARS[1] + 0.5,
               color="none", ec="#666666", ls=":", lw=1.0)
    ax.plot(sp.year, sp.wb_spei, "-o", ms=3.4, lw=1.3, color="#d95f02",
            label="World Bank annual SPEI (EN.CLC.SPEI.XD)")
    ax.plot(sp.year, sp.ours_spei6_mean, "-s", ms=3.4, lw=1.3, color="#2c7fb8",
            label="ours: mean district SPEI6 (ERA5)")
    ax.plot(sp.year, sp.ours_spei6_min, "^", ms=4, color="#9ecae1",
            label="ours: driest district-month SPEI6")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(-1, color="#b30000", lw=0.9, ls="--")
    ax.text(int(sp.year.min()), -1.06, "SPEI $=-1$ (severe)", fontsize=7,
            color="#b30000", va="top")
    ax.set_xlim(1989, 2024)
    ax.set_ylim(-5.2, 3.0)
    ax.set_xlabel("year")
    ax.set_ylabel("standardised index")
    ax.set_title("Independent national drought index vs the one derived here\n"
                 f"shared years {a['years'][0]}-{a['years'][-1]}: "
                 f"r={a['pearson_r']:.2f} (p={a['pearson_p']:.2f}), "
                 f"sign agreement {a['sign_agreement']:.0%}", fontsize=9)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#e8b4b8", alpha=0.45,
                         label="register drought year"))
    handles.append(Line2D([0], [0], color="#666666", ls=":",
                          label=f"ERA5 panel {C.EVAL_TRAIN_YEARS[0]}-"
                                f"{C.EVAL_TEST_YEARS[1]}"))
    ax.legend(handles=handles, fontsize=6.8, loc="lower left", ncol=2,
              columnspacing=0.9, handletextpad=0.4)

    ax = axes[1]
    yrs = pr.year.values
    ax.bar(yrs, pr.era5_mean_mm, width=0.75, color="#74a9cf",
           label="ERA5 district-mean annual precipitation")
    ax.axhline(pc["wb_normal_mm"], color="#a63603", lw=1.4, ls="--",
               label=f"World Bank normal {pc['wb_normal_mm']:.0f} mm "
                     f"({pc['wb_source'] or 'WDI/FAO'})")
    dry_site, dry_mm = pc["era5_driest_site"]
    wet_site, wet_mm = pc["era5_wettest_site"]
    ax.axhspan(dry_mm, wet_mm, color="#e5e5e5", alpha=0.7, lw=0, zorder=0,
               label="site-mean range across the 16 districts")
    ax.axhline(dry_mm, color="#999999", lw=0.9, ls=":")
    ax.axhline(wet_mm, color="#999999", lw=0.9, ls=":")
    ax.text(yrs.min() - 0.4, wet_mm + 120, f"wettest site {wet_site} "
            f"{wet_mm:.0f} mm", fontsize=6.5, color="#555555", va="bottom")
    ax.text(yrs.min() - 0.4, dry_mm - 120, f"driest site {dry_site} "
            f"{dry_mm:.0f} mm", fontsize=6.5, color="#555555", va="top")
    ax.set_xlabel("year")
    ax.set_ylabel("mm / yr")
    ax.set_xlim(yrs.min() - 1.0, yrs.max() + 1.0)
    ax.set_ylim(0, wet_mm * 1.12)
    ax.set_title("Reanalysis level check\n"
                 f"mean {pc['era5_mean_annual_mm']:.0f} mm = "
                 f"{pc['pct_of_normal']:.0f}% of the Bank's normal; "
                 f"{pc['era5_complete_year_loc_rows']} complete district-years",
                 fontsize=9)
    ax.legend(fontsize=6.2, loc="upper right")

    fig.suptitle("World Bank Climate Portal cross-check: a national annual index "
                 "that does not track the register's drought years",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    C.save_figure(fig, "fig_wb_check")
    print(f"fig_wb_check: {len(sp)} SPEI years, {len(pr)} precipitation years, "
          f"{len(dr)} shaded drought years, r={a['pearson_r']}")


if __name__ == "__main__":
    main()
