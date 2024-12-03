#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Define parameters
params.input_csv = "${projectDir}/input/data.csv"
params.preprocess_script = "${projectDir}/scripts/preprocess_data.py"
params.split_script = "${projectDir}/scripts/split_dataset.py"
params.rf_script = "${projectDir}/scripts/random_forest_prediction.py"
params.visualize_script = "${projectDir}/scripts/visualize_confusion_matrix.py"
params.output_dir = "./output"
params.seed = 42

/*
 * Main workflow definition
 */
workflow {
    // Channels for input data and scripts
    input_csv_ch = Channel.fromPath(params.input_csv)
    preprocess_script_ch = Channel.fromPath(params.preprocess_script)
    split_script_ch = Channel.fromPath(params.split_script)
    rf_script_ch = Channel.fromPath(params.rf_script)
    visualize_script_ch = Channel.fromPath(params.visualize_script)

    // Step 1: Preprocess data
    preprocessed_data_ch = preprocessData(input_csv_ch, preprocess_script_ch)

    // Step 2: Split dataset
    split_output = splitDataset(preprocessed_data_ch, split_script_ch)

    // Step 3: Random Forest prediction
    rf_results = randomForestPrediction(split_output, rf_script_ch)

    // Step 4: Visualize confusion matrix
    confusion_matrix_ch = rf_results.map { it[1] } // Extract confusion_matrix.npy
    visualizeConfusionMatrix(confusion_matrix_ch, visualize_script_ch)
}

/*
 * Process: Preprocess Data
 */
process preprocessData {

    input:
    path input_csv
    path script_file

    output:
    path "preprocessed_data.csv"

    container 'selinasun01/phaseprediction:latest'

    script:
    """
    mkdir -p ${params.output_dir}
    python ${script_file} --input ${input_csv} --output preprocessed_data.csv
    cp preprocessed_data.csv ${params.output_dir}/preprocessed_data.csv
    """
}

/*
 * Process: Split Dataset
 */
process splitDataset {

    input:
    path preprocessed_file
    path script_file

    output:
    tuple path("train.csv"), path("valid.csv"), path("test.csv")

    container 'selinasun01/phaseprediction:latest'

    script:
    """
    mkdir -p ${params.output_dir}
    python ${script_file} --input ${preprocessed_file} --output_dir . --seed ${params.seed}
    cp train.csv ${params.output_dir}/train.csv
    cp valid.csv ${params.output_dir}/valid.csv
    cp test.csv ${params.output_dir}/test.csv
    """
}

/*
 * Process: Random Forest Prediction
 */
process randomForestPrediction {

    input:
    tuple path(train_file), path(valid_file), path(test_file)
    path script_file

    output:
    tuple path("results/metrics.txt"), path("results/confusion_matrix.npy")

    container 'selinasun01/phaseprediction:latest'

    script:
    """
    mkdir -p results
    python ${script_file} \
        --train ${train_file} \
        --valid ${valid_file} \
        --test ${test_file} \
        --metrics results/metrics.txt \
        --confusion_matrix results/confusion_matrix.npy

    mkdir -p ${params.output_dir}/results
    cp results/metrics.txt ${params.output_dir}/results/metrics.txt
    cp results/confusion_matrix.npy ${params.output_dir}/results/confusion_matrix.npy
    """
}

/*
 * Process: Visualize Confusion Matrix
 */
process visualizeConfusionMatrix {

    input:
    path confusion_matrix
    path script_file

    output:
    path "${params.output_dir}/results/confusion_matrix_label_*.png"

    container 'selinasun01/phaseprediction:latest'

    script:
    """
    mkdir -p results
    python ${script_file} \
        --input ${confusion_matrix} \
        --output results/confusion_matrix.png

    mkdir -p ${params.output_dir}/results
    cp results/confusion_matrix_label_*.png ${params.output_dir}/results/
    """
}



