#!/bin/sh
# Renames the figures that the report no longer uses, adding the suffix _to_be_removed_N.
# Run from this folder:  sh rename_to_be_removed.sh
# Nothing is deleted; the renamed files can be removed by hand once checked.
cd "$(dirname "$0")"

ren() { [ -f "$1" ] && mv -n "$1" "${1%.png}_to_be_removed_$2.png" && echo "renamed $1"; }

# Replaced by the P=S=16 figures of Appendix G (FIG_G_*.png)
ren FIG_E3_kernelsynth_p16-s12_main_components_seed59.png 1
ren FIG_E4_kernelsynth_p16-s12_spectra_seed59.png 2
ren FIG_E3_tsmixup_p16-s12_main_components_seed74.png 3
ren FIG_E4_tsmixup_p16-s12_spectra_seed74.png 4

# Also not referenced by any section of the report. Uncomment to rename them too.
# ren FIG_APPENDIX_F_depth_profile.png 5
# ren FIG_APPENDIX_F_gap_bypatch.png 6
# ren FIG_APPENDIX_F_gap_bystride.png 7
# ren FIG_E3_kernelsynth_p16-s16_main_components_seed2.png 8
# ren FIG_E3_kernelsynth_p16-s16_main_components_seed33.png 9
# ren FIG_E3_tsmixup_p16-s16_main_components_seed17.png 10
# ren FIG_E3_tsmixup_p16-s16_main_components_seed85.png 11
# ren FIG_E4_kernelsynth_p16-s16_spectra_seed2.png 12
# ren FIG_E4_kernelsynth_p16-s16_spectra_seed33.png 13
# ren FIG_E4_tsmixup_p16-s16_spectra_seed17.png 14
# ren FIG_E4_tsmixup_p16-s16_spectra_seed2.png 15
# ren FIG_E4_tsmixup_p16-s16_spectra_seed85.png 16
# ren FIG_SV_PAIRED_p16-s16.png 17
# ren FIG_SV_published_p16-s16.png 18
# ren solar_forecast_day186_0545_both.png 19
# ren solar_forecast_day186_0545_horizon.png 20
# ren solar_forecast_day186_1300_bypatch.png 21
# ren solar_forecast_day186_1300_bystride.png 22
# ren solar_forecast_day186_1300_p16-s16_error.png 23
# ren solar_forecast_day186_1700_both.png 24
# ren solar_forecast_day186_1700_error.png 25
