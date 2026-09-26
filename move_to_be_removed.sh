#!/usr/bin/env bash
# Moves the notebooks, figures and data that Deliverable 3 does not use into to_be_removed/,
# keeping each file's path relative to the repository root, so that it can be reviewed and then
# deleted in one step (rm -rf to_be_removed). Tracked files are moved with git mv, so the removal
# shows up in git as a rename that the later delete turns into a deletion.
#
# Run from the repository root:   bash move_to_be_removed.sh
set -euo pipefail
cd "$(dirname "$0")"
DEST="to_be_removed"

move() {
  local src="$1"
  if [ ! -e "$src" ]; then echo "  skipped (absent): $src"; return; fi
  mkdir -p "$DEST/$(dirname "$src")"
  if git ls-files --error-unmatch "$src" >/dev/null 2>&1 || \
     [ -n "$(git ls-files "$src" 2>/dev/null)" ]; then
    git mv "$src" "$DEST/$src"
  else
    mv "$src" "$DEST/$src"
  fi
  echo "  moved: $src"
}

echo "Notebooks not used by Deliverable 3"
move notebooks/signal_analysis/solar_telemetry_benchmark_colab.ipynb   # superseded G_h benchmark
move notebooks/signal_analysis/appendix_decomposition.ipynb            # single-model version, superseded by appendix_decomposition_comparison
move notebooks/data/synthetic/signals_exploration.ipynb                 # old inject API, fails with the current generators

echo "Figures not referenced by the report"
F=coursework/deliverable3/figures
for f in \
  FIG_APPENDIX_F_depth_profile.png FIG_APPENDIX_F_gap_bypatch.png FIG_APPENDIX_F_gap_bystride.png \
  FIG_E3_kernelsynth_p16-s16_main_components_seed2.png FIG_E3_kernelsynth_p16-s16_main_components_seed33.png \
  FIG_E3_tsmixup_p16-s16_main_components_seed17.png FIG_E3_tsmixup_p16-s16_main_components_seed85.png \
  FIG_E4_kernelsynth_p16-s16_spectra_seed2.png FIG_E4_kernelsynth_p16-s16_spectra_seed33.png \
  FIG_E4_tsmixup_p16-s16_spectra_seed17.png FIG_E4_tsmixup_p16-s16_spectra_seed2.png \
  FIG_E4_tsmixup_p16-s16_spectra_seed85.png \
  FIG_SV_PAIRED_p16-s16.png FIG_SV_published_p16-s16.png \
  solar_forecast_day186_0545_both.png solar_forecast_day186_0545_horizon.png \
  solar_forecast_day186_1300_bypatch.png solar_forecast_day186_1300_bystride.png \
  solar_forecast_day186_1300_p16-s16_error.png solar_forecast_day186_1700_both.png \
  solar_forecast_day186_1700_error.png; do
  move "$F/$f"
done

echo "Deliverable 1 signal archive (l_syn = 576), not read by any current script"
move notebooks/data/synthetic/signals

echo
echo "Done. Review $DEST/, then delete it with:  rm -rf $DEST   (and git commit)"
