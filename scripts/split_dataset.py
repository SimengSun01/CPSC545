import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_file, output_dir, seed):
    df = pd.read_csv(input_file)

    #separate features and labels
    X = df.iloc[:, :-1]  # feature data 
    y = df.iloc[:, -1]   # (label)

    #split into train, temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)

    #split temp into validation and test
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    # Save datasets to CSV
    pd.concat([X_train, y_train], axis=1).to_csv(f"{output_dir}/train.csv", index=False)
    pd.concat([X_valid, y_valid], axis=1).to_csv(f"{output_dir}/valid.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(f"{output_dir}/test.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, valid, and test sets.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    split_dataset(args.input, args.output_dir, args.seed)
