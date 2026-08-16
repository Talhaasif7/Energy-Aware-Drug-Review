# WebMD External Validation Dataset

Due to GitHub's **100 MB single file limit**, `webmd.csv` (160.64 MB, 320,096 cleaned reviews) is excluded from Git tracking in `.gitignore`.

### Obtaining `webmd.csv`:
1. Download the raw `webmd.csv` from Kaggle: [WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/jessicali9530/webmd-drug-reviews-dataset).
2. Place `webmd.csv` in this folder (`data/02_secondary_sentiment_scaling/external_val_webmd/`).
3. Run `python scripts/refine_datasets.py` to clean and standardize the dataset.
