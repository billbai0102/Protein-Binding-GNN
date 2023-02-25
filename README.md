# Protein-Binding-GNN
From Cyclica's AF2 Dataset

##### How to replicate the test results:
1. Download LSTM embeddings from https://drive.google.com/file/d/1hXDXvksjqoI3hS2i_pyOp_Lw6hTMzS-E/view?usp=sharing and place it in ./data/
2. Download test csv w/o labels and place it in ./data/
2. Run generate_test_results_ipynb
3. Submissions will be generated in ./submission.csv

##### How to replicate training:
1. Download train file from cyclica and place in ./data2/raw/, and download AlphaFold2 Human proteins dataset directly from the website and place in ./data/
2. Run train_model.ipynb 
3. Takes about 5 minutes on a 3090ti 

If you need help ping me on Discord @X5#7495

##### Files:
**LigandGNNV2.py** - The Graph Neural Network \
**train_model.ipynb** - The notebook where the model is trained in \
**EmbedDataset.py** - The Dataset class that generates the graphs from the csv \
**generate_test_results.ipynb** - The notebook that generates the submission csv \
