# Week 14 Activities: HW7 Streamflow forecast Repo Setup


## Activity 1 — Set up your repository

**1. Create a new GitHub repository for HW7**

- Go to [github.com](https://github.com) and sign in to your account.
- Click the **+** menu (top right) → **New repository**.
- Name the repository `Analysis_HW7`. Leave it **Public** 
- Click **Create repository**.

**2. Clone your repo into the class Codespace**

Navigate to the class Codespace. In the terminal, clone your new repository using the `git clone` command from terminal. Make sure you change directories to a location you want to put it before you clone it. 

**3. Copy the starter scripts**

Grab the starter files from the `week14` folder of the class repository and copy them into your `Analysis_HW7` directory.

**4. Make the shell script executable**

Before the workflow script can be run, you need to give the shell permission to execute it. From within a terminal change directories (`cd`) to the directory with the scripts you just copied and then run run the following command to make the script executable:

```bash
chmod +x run_workflow.sh
```

You only need to do this once. After you do this if you do a ls in the directory  you should see the `run_workflow.sh` file has changed colors. 

**5. Commit and push to your repo**

Verify the files appear on your GitHub repository page before moving on.

---

## Activity 2 — Create a conda environment and `environment.yml`

**1. Identify the required packages**

Open the starter scripts and look through the `import` statements. Make a list of the third-party packages that will need to be installed (ignore standard library modules like `os`, `argparse`, and `pickle`). Note that `hf_hydrodata` is not available on conda and will need to be installed via `pip`. This means you will also need to conda install `pip`.

**2. Create the environment and install packages**

```bash
conda create -n hw7_forecast python=3.11
conda activate hw7_forecast
conda install <packages>
pip install <pip-only package>
```

**3. Export the `environment.yml`**

Use `--from-history` to export only the packages you explicitly installed, rather than the full resolved dependency tree:

```bash
conda env export --from-history > environment.yml
```

Open the file and check that it looks reasonable — it should list just the packages you asked for, not hundreds of internal dependencies.

**4. Test it**

To truly test the `environment.yml`, you need to remove the environment you just built and recreate it from the file. First deactivate and delete it, then rebuild from scratch:

```bash
conda deactivate
conda env remove -n hw7_forecast
conda env create -f environment.yml
conda activate hw7_forecast
```

If the environment activates and your scripts run correctly, your `environment.yml` is complete.

Don't forget to commit and push this change to your repo before you go on. 

**5. Adding a package later**

If you discover a missing package, install it into the active environment and regenerate the file:

```bash
conda install <new-package>        # or: pip install <new-package>
conda env export --from-history > environment.yml
```

Then commit the updated file so your environment stays reproducible.

---

## Activity 3 — Map the workflow

The class will split into two groups. Both groups start from `run_workflow.sh` and trace the full execution path of their assigned script.

- **Group 1** — `train_model.py`
- **Group 2** — `generate_forecast.py`

Draw a diagram that shows:

1. **The entry point** — which variables/arguments does `run_workflow.sh` define and pass to your script?
2. **The script flow** — what does the script do step by step?
3. **Every function called** — for each function imported from `forecast_functions.py`, show:
   - What it does (one sentence)
   - Its input arguments
   - What it returns
4. **Data inputs and outputs** — what external data is downloaded, what files are read from disk, and what files are written?

Your diagram can be a flowchart, a box-and-arrow sketch, or any format that clearly shows the relationships. 

---

## Activity 4 — Add the monthly average model

The starter code includes a long-term average model that predicts the same value — the mean streamflow over the entire training period — for every forecast day. Your task is to add a **monthly average model** that is slightly smarter: instead of one global mean, it uses a separate historical mean for each calendar month.

**Files you will need to modify**

- `forecast_functions.py`
- `train_model.py`
- `generate_forecast.py`
- `run_workflow.sh`

**Before you start**
Make sure that you can run the workflow in its current form. To run it just type `./run_workflow.sh` from a terminal window in the folder where your runscript is.  

**Step 1 — Add the model functions to `forecast_functions.py`**

You need to write two functions, following the same pattern as the `longterm_avg` equivalents already in the file.

*Fitting function* — `fit_monthly_avg_model(train_df)` should return a dictionary mapping each calendar month (1–12) to the mean streamflow for that month over the training period. The following snippet will be useful:

```python
train_df.groupby(train_df.index.month)['streamflow_cfs'].mean().to_dict()
```

*Forecast function* — `make_5day_forecast_monthly(monthly_means, forecast_date, n_days=5)` should return a DataFrame with `Forecast_cfs` indexed by date, where each day's forecast is the historical mean for that day's calendar month. To look up the month for a date `d`, use `d.month`. You can also look it up for a series of dates like this:

```python
[monthly_means[d.month] for d in dates]
```

**Step 2 — Add the model branch to `train_model.py`**

- Import `fit_monthly_avg_model` at the top of the file alongside the other imports.
- Add `'monthly_avg'` to the `--model` argument choices.
- Add an `elif args.model == 'monthly_avg':` block. Use the `longterm_avg` block directly above it as your template — the structure is identical, you just need to swap in the new functions and adjust the printed labels.
  - Hint: For the vailidation block you can get the timeseries you need like this: 
  ```python
          train_fitted = pd.Series([monthly_means[d.month] for d in train.index],index=train.index
        )
        forecast_series = pd.Series([monthly_means[d.month] for d in test.index],index=test.index
        )
   ```

- After you do this you can test running just the model fitting part of the workflow.  Add `monthly_avg` as an option in the comment next to the `MODEL` variable, then set `MODEL="monthly_avg"`, comment out the generate_forecast step and run the workflow to test your implementation. 

**Step 3 — Add the model branch to `generate_forecast.py`**

- Import `make_5day_forecast_monthly` at the top of the file.
- Add `'monthly_avg'` to the `--model` argument choices.
- Add an `elif args.model == 'monthly_avg':` block, again using `longterm_avg` as your template.
   - Hint: When you copy the part that is checking if the model exists you need to set the type to 'dict' because that is how we trained it. You can do that like this `if not isinstance(monthly_means, dict):`
- Run the whole workflow now with the generate forecast part uncommented. 

**Step 5 — Commit and push your changes**

