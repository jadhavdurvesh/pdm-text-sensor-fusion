# PdM Fusion Project — Step 1: Sensor-Only Baseline

This is the first working piece of your FYP: a sensor-only LSTM that predicts
Remaining Useful Life (RUL) on the NASA C-MAPSS FD001 dataset. This becomes
your **baseline number** — later, your text-fusion model needs to beat it.

## 1. Get the dataset

Download the C-MAPSS Turbofan Degradation dataset (search "NASA C-MAPSS
dataset" — it's hosted on the NASA Prognostics Data Repository / Kaggle
mirrors). You need these three files for FD001:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

Place them in a `data/` folder next to these scripts:

```
pdm_project/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── data_utils.py
├── model.py
├── train_baseline.py
├── requirements.txt
└── README.md
```

## 2. Set up your environment

**Recommended: Google Colab** (free GPU, nothing to install):
1. Upload all files in this folder to your Colab session (or clone from your
   GitHub repo).
2. Upload the 3 data files into a `data/` folder there too.
3. Run: `!pip install -q -r requirements.txt` (torch is pre-installed on
   Colab, so this is mostly for `transformers`, needed later for the fusion
   step).

**Local machine:**
```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 3. Verify the data pipeline works

Before training, run the built-in smoke test — it uses synthetic dummy data
to confirm the preprocessing code runs correctly on your machine, with no
real dataset required yet:

```bash
python data_utils.py
```

You should see `Smoke test passed.`

## 4. Train the baseline

Once your real data files are in `data/`:

```bash
python train_baseline.py
```

This will:
- Load and normalize the FD001 sensor data
- Compute RUL labels (capped at 125, standard practice for this dataset)
- Slice into sliding windows of 30 cycles
- Train an LSTM for 40 epochs
- Print validation RMSE per 5 epochs, then a final **test RMSE**
- Save the trained model to `baseline_lstm.pt`

**Write down the final test RMSE** — this is your baseline result for the
paper's results table. Typical published RMSE values on FD001 with a simple
LSTM are roughly in the 15–25 range; don't worry if you're not immediately
competitive with state-of-the-art papers that use heavier architectures —
your contribution is the fusion comparison, not beating the sensor-only SOTA.

## 5. Generate synthetic maintenance logs

Once your baseline has run successfully and you have your RMSE (e.g. 13.121),
generate the synthetic technician-log text that the fusion model will use:

```bash
python synthetic_logs.py
```

This will:
- Reload and re-window the same C-MAPSS data (same windows your baseline used)
- For each window, compare the start vs. end of the window per sensor to find
  the biggest shifts, using real C-MAPSS sensor physics (temperature,
  pressure, speed, flow — see `SENSOR_INFO` in `synthetic_logs.py`)
- Write a technician-style note whose severity (routine / minor deviation /
  urgent) scales with how close the engine is to failure (RUL)
- Save two files:
  - `results/train_logs.csv` (columns: `RUL`, `log_text`)
  - `results/test_logs.csv` (same columns)

It prints 3 sample logs at the end so you can sanity-check the output looks
reasonable before moving on. It's deterministic (fixed random seed), so
re-running it gives you the same logs every time — important for your
reproducibility section.

**Nothing to download for this step** — it only needs the C-MAPSS data
already in `data/` from step 1.

## 6. Train the fusion model

Once `results/train_logs.csv` and `results/test_logs.csv` exist, train the
fusion model (LSTM sensor encoder + DistilBERT text encoder):

```bash
python train_fusion.py --baseline_rmse 13.121
```

(Replace `13.121` with your own baseline test RMSE from step 4.)

This will:
- Load the same sensor windows as your baseline, plus the synthetic logs
- Download `distilbert-base-uncased` on first run (needs internet — use
  Colab or a machine with a connection; it's cached after that)
- Train a fusion model that combines both sensor and text signals
- By default the text encoder (DistilBERT) is **frozen** — only the
  projection layer and fusion head are trained. This is much faster and a
  sensible first run; you can set `FREEZE_TEXT_ENCODER = False` at the top
  of `train_fusion.py` later for full fine-tuning if you have time/compute
  for an ablation comparison
- Print a final **Baseline vs. Fusion comparison table** and save it to
  `results/comparison.csv`
- Save the trained model to `fusion_model.pt`

**This comparison table is your core paper result.** Whether the fusion
model improves on the baseline or not, both outcomes are reportable —
if it doesn't improve, that's still a valid, discussable finding (e.g., the
synthetic logs may need richer variation, or the fusion method may need
attention instead of concatenation — good material for your Discussion/
Limitations section).

## 7. What's next

Once you have your baseline RMSE, fusion RMSE, and comparison table, you're
ready to write up the Results and Discussion sections of your paper. Come
back and I can help you:
- Interpret and write up the results
- Draft the Discussion/Limitations section
- Put together the full paper structure for submission
