import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(input_file, output_file):
    """
    Preprocesses the input CSV file by normalizing numeric data
    and get the label column.
    """
    ####csv input 
    df = pd.read_csv(input_file)
    #separate features and labels
    label_column = df.columns[-1]  #the last column is labels 
    features = df.iloc[:, :-1]    #all columns except the last one (features)
    labels = df[label_column]    #last column as labels

    #normalize numeric  using min max scaler 
    #scanpy log1p doesnt perform better

    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    #create normalized DataFrame
    normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
    normalized_df[label_column] = labels

    # save 
    normalized_df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess input CSV file and normalize numeric features.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to save preprocessed data.")

    args = parser.parse_args()
    preprocess_data(args.input, args.output)

