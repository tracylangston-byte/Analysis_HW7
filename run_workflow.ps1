# run_workflow.ps1
# Windows PowerShell version of run_workflow.sh

# =============================================================================
# USER OPTIONS — edit these before running
# =============================================================================

$GAUGE_ID = "09506000"      # USGS gauge ID
$AR_ORDER = 7               # Number of lag days for the AR model

$TRAIN_START = "1990-01-01"
$TRAIN_END = "2022-12-31"
$TEST_START = "2023-01-01"
$TEST_END = "2024-12-31"

$FORECAST_DATE = "2024-04-30"   # First day of the 5-day forecast
$REFIT_MODEL = "True"           # True = re-fit from scratch | False = use saved_model.pkl
$RUN_VALIDATION = "True"        # True = show validation plots and metrics
$MODEL = "longterm_avg"         # longterm_avg = training mean

# =============================================================================
# RUN WORKFLOW — no need to edit below this line
# =============================================================================

$EMAIL = Read-Host "HydroFrame email"
$PIN_SECURE = Read-Host "HydroFrame PIN" -AsSecureString

# Convert hidden PIN input back to plain text so Python can receive it
$PIN = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($PIN_SECURE)
)

if ($REFIT_MODEL -eq "True" -or $RUN_VALIDATION -eq "True") {
    python train_model.py `
        --email $EMAIL `
        --pin $PIN `
        --gauge-id $GAUGE_ID `
        --ar-order $AR_ORDER `
        --train-start $TRAIN_START `
        --train-end $TRAIN_END `
        --test-start $TEST_START `
        --test-end $TEST_END `
        --model $MODEL `
        --refit $REFIT_MODEL `
        --validate $RUN_VALIDATION
}

python generate_forecast.py `
    --email $EMAIL `
    --pin $PIN `
    --gauge-id $GAUGE_ID `
    --ar-order $AR_ORDER `
    --forecast-date $FORECAST_DATE `
    --model $MODEL