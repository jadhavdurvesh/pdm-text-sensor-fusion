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

## 5. What's next

Once this baseline is working and you have your RMSE number, the next steps
are:
1. Generate synthetic maintenance-log text tied to the failure patterns in
   this same data.
2. Build the fusion model (this baseline's LSTM + a BERT text encoder).
3. Compare fusion RMSE against this baseline RMSE — that comparison is your
   core paper result.

Come back and I'll help you build the synthetic log generator next.
