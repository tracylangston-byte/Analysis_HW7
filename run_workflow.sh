#!/bin/bash
# run_workflow.sh
# ---------------
# DRIVER SCRIPT — set your options here and run:
#
#   bash run_workflow.sh

# =============================================================================
# USER OPTIONS — edit these before running
# =============================================================================

GAUGE_ID="09506000"   # USGS gauge ID (Verde River near Camp Verde, AZ)
AR_ORDER=7            # Number of lag days for the AR model

TRAIN_START="1990-01-01"
TRAIN_END="2022-12-31"
TEST_START="2023-01-01"
TEST_END="2024-12-31"

FORECAST_DATE="2024-04-30"   # First day of the 5-day forecast (YYYY-MM-DD)
REFIT_MODEL="True"           # True = re-fit from scratch | False = use saved_model.pkl
RUN_VALIDATION="True"        # True = show validation plots and metrics
MODEL="longterm_avg"         # longterm_avg = training mean

# =============================================================================
# RUN WORKFLOW — no need to edit below this line
# =============================================================================

read -p  "HydroFrame email: " EMAIL
read -sp "HydroFrame PIN:   " PIN && echo

if [ "$REFIT_MODEL" = "True" ] || [ "$RUN_VALIDATION" = "True" ]; then
    python train_model.py \
        --email        "$EMAIL"       \
        --pin          "$PIN"         \
        --gauge-id     "$GAUGE_ID"    \
        --ar-order     "$AR_ORDER"    \
        --train-start  "$TRAIN_START" \
        --train-end    "$TRAIN_END"   \
        --test-start   "$TEST_START"  \
        --test-end     "$TEST_END"    \
        --model        "$MODEL"       \
        --refit        "$REFIT_MODEL" \
        --validate     "$RUN_VALIDATION"
fi

python generate_forecast.py \
    --email         "$EMAIL"         \
    --pin           "$PIN"           \
    --gauge-id      "$GAUGE_ID"      \
    --ar-order      "$AR_ORDER"      \
    --forecast-date "$FORECAST_DATE" \
    --model         "$MODEL"
