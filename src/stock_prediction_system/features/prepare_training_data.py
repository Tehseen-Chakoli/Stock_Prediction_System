from pathlib import Path
import pandas as pd

PROCESSED_DATA_DIR = Path("data/processed")


def load_features():
    df = pd.read_csv(PROCESSED_DATA_DIR / "sbi_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_training_data(df: pd.DataFrame):
    df = df.copy()

    # Drop rows with ANY NaNs
    df = df.dropna().reset_index(drop=True)

    return df


def save_training_data(df: pd.DataFrame):
    output_path = PROCESSED_DATA_DIR / "sbi_training_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved training data at {output_path}")


def run():
    df = load_features()
    clean_df = prepare_training_data(df)

    print("Final Training Shape:", clean_df.shape)

    print("\nPreview:")
    print(clean_df.head())

    save_training_data(clean_df)


if __name__ == "__main__":
    run()