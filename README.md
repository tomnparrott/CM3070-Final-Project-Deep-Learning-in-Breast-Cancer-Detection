# CM3070 Final Project

Final year project. Machine Learning Breast Cancer template. My project uses ResNet-18 with transfer learning to classify mammography images as malignant or benign.

## What you need to run the code

Python 3.10 or higher. I used an RTX 3070 GPU but it will run on CPU if you don't have one.

Install the dependencies with:

    pip install -r requirements.txt

To use the Streamlit dashboard you also need to install:

    pip install streamlit

## About the dataset

The raw images come from CBIS-DDSM on The Cancer Imaging Archive (https://www.cancerimagingarchive.net/collection/cbis-ddsm/). The full dataset is around 180GB of DICOM files so it isn't included here. If you want to rerun training from scratch you would need to download it from TCIA and put it in Data/downloaded data/.

I have included the outputs from my best training run in Data/Processed/runs/20260223_185850_seed42/ so you can see all the results (metrics, figures, Grad-CAM overlays, model checkpoints) without needing the raw images.

The CSV files in Data/Processed/ (manifest.csv, splits.csv, train.csv, val.csv, test.csv) show the full dataset makeup and how it was split.

## How to run the pipeline

There are four scripts that run in sequence. Each one picks up where the last one left off by reading files from disk.

    python -m SRC.cbis_manifest
    python -m SRC.split_data
    python -m SRC.train
    python -m SRC.eval

cbis_manifest.py reads the CBIS-DDSM metadata CSVs and builds a single manifest.csv that maps each patient to their images and labels. split_data.py takes that manifest and does a patient level split into train/val/test (70/15/15) so that no patient appears in more than one set, this mitigates the risk of data leakage, as detailed in the report; train.py trains the model and saves the best checkpoint based on validation AUC; eval.py loads that checkpoint and runs it on the test set, producing all the metrics, plots, confusion matrices, calibration curves, subgroup breakdowns and Grad-CAM failure analysis.

Everything gets saved into a timestamped folder under Data/Processed/runs/. Each run folder has the checkpoint, a copy of the config that was used, and an eval_outputs folder with all the figures and JSON metrics.

To look at the results in a browser:

    streamlit run dashboard.py

This loads the saved evaluation outputs and lets you compare different runs side by side.

## Config

All the settings are in Configs/config.yaml. Image size, learning rate, number of epochs, whether augmentation is on, class weighting, evaluation thresholds etc. The config gets copied into each run folder so you can check what settings produced a given set of results for reproducibility.

## Code layout

SRC/ has all the main code. cbis_manifest.py, split_data.py, dataset.py, model.py, train.py, eval.py, inference.py. There's also paths.py, seed.py and case_index.py for shared utilities.

Configs/ has the YAML config file.

tests/ has the pytest tests. Some of them need a completed training run to work so they will skip if the outputs aren't there. Run them with pytest tests/ -v.

tools/ has audit_summary.py which summarises the Grad-CAM audit flags.

dashboard.py is the Streamlit app.
