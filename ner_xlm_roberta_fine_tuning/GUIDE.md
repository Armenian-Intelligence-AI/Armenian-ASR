# NER Model Fine-Tuning and Inference


### 1. Preprocess Raw Data (`preprocess_raw_data.ipynb`)

This notebook preprocesses raw `.txt` files and converts them into a format suitable for model training. It handles the initial cleaning and structuring of the data.

- **Input**: Raw `.txt` files containing text data.
- **Output**: Preprocessed files ready for further formatting.

### 2. Preprocess and Create Folder with Text Files (`preprocess_and_create_folder_with_txt_files.ipynb`)

This notebook continues from where the first preprocessing step ends. It further processes the output from `preprocess_raw_data.ipynb`, organizing the files into folders and structuring them for dataset creation.

- **Input**: Output from the first preprocessing step.
- **Output**: Well-structured folders containing text files, ready for dataset creation.

### 3. Create Dataset (`make_dataset.ipynb`)

This notebook is responsible for creating a dataset from the preprocessed text files. It organizes the data into the correct format for model training.

- **Input**: Preprocessed text files.
- **Output**: A final dataset in the required format for NER model training.

### 4. Fine-Tune and Inference (`main.ipynb`)

This is the main notebook for fine-tuning the NER model on the dataset created in the previous step. It includes code for training the model as well as running inference on test data to evaluate the model's performance.

- **Input**: Dataset from the `make_dataset.ipynb` notebook.
- **Output**: A fine-tuned NER model and the results from inference tests.

## How to Use

1. **Step 1**: Run `preprocess_raw_data.ipynb` to clean and structure the raw text data.
2. **Step 2**: Execute `preprocess_and_create_folder_with_txt_files.ipynb` to further preprocess the data and organize it into folders.
3. **Step 3**: Run `make_dataset.ipynb` to create the dataset for training.
4. **Step 4**: Use `main.ipynb` to fine-tune the model and run inference to evaluate the results.

## Requirements

- Python 3.x
- Jupyter Notebook
- Necessary Python packages

## Notes

- Ensure all data is correctly formatted and placed in the appropriate directories before starting the workflow.
- Adjust hyperparameters in `main.ipynb` based on the specific requirements of your model.