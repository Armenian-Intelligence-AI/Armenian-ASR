Fine-Tuning Guide

Step 1: Preprocess the Data

Check the asr_data_preprocessing.ipynb for details

Step 2: Create the Dataset

To create the dataset, place the metadata.csv file in the same directory as your wavs folder. The dataset creation script expects this structure:

/dataset/
   ├── wavs/
   └── metadata.csv

Run the dataset creation script:

python create_dataset.py

	•	The script will automatically create a dataset/ folder containing the necessary files for fine-tuning.

Step 3: Fine-Tune the Model

Finally, run the fine_tune.py script to train the model using the dataset generated in the previous step:

python fine_tune.py

	•	Input: The dataset/ folder created in the previous step.
	•	Output: The fine-tuned model.

