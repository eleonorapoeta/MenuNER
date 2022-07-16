# MenuNER
This work is done on the paper 
'MenuNER: Domain-Adapted BERT Based NER Approach for a
Domain with Limited Dataset and Its Application to Food
Menu Domain' 
( https://www.mdpi.com/2076-3417/11/13/6007 ).
We started implementing from scratch all the methods present in the paper 
and then performing some additional experiments.
All the experiments and the method details can be found in the report file.


## Abstract

<i>Named entity recognition (NER) is a sub-task of information extraction that seeks to locate and classify named entities in text, into pre-defined categories such as the names of persons, organizations, locations, etc.
In this paper, we focus on Menu entity extraction from online user reviews for the restaurant.
The approach involves first adapting NER task on a new domain where large datasets are rarely available,
embedding and fine-tuning the network model BiLSTM with CRF, very popular in NER tasks, using extended
feature vectors.</i>

## Our Architecture

<p align="center">
<img src="https://github.com/eleonorapoeta/MenuNER/images/architecture.jpg" alt="architeture" width="70%"/>
</p>
MenuNER architecture used in this work. Each word of each sentence is embedded with POS, BERT and Char 
embedding layer (green). Then the BiLSTM (orange), along with the 
MultiHeadAttention (cyan), the Linear layer(lilac) and the CRF module 
(turquoise) work to predict the label of the word between 'O' and 'MENU'.

## Setup
First, clone the GitHub and create a new virtual environment in the project directory and activate it.
```python
git clone https://github.com/eleonorapoeta/MenuNER/
python3 -m venv .env
source .env/bin/activate
```
Then, install all requirements
```python
pip install -r requirements.txt
```

## Reviews Retrieval and manipulation
After the setting the virtual environment and installation of the requirements,
to perform this part of the project you need to search 
for the Yelp Dataset available at
 https://www.yelp.com/dataset/ .

```python
cd ./MenuNER/bert_training
reviews.py --reviews_filename --business_filename 
```
## BERT domain
To train the BERT domain on the reviews extracted, you can also add a previous 
BERT checkpoint from which you can resume the training
```python
bert_domain.py --filtered_reviews_filename  --checkpoint_path 
```
## Training 
You can train the BiLSTM_CRF model 
with the possibility of choosing, for each module, 
whether it contributes or not. Train and Validation files are provided.
```python
cd ./MenuNER/ner
train.py 
--data_dir 
--seed 
--max_length 
--train 
--val
--lr
--epochs
--bert_checkpoint
--pos
--pos_embedding_dim
--char
--char_embedding_dim
--attention
--eval_and_save_steps
```


## Test
You can test your trained model on the test 
file already provided.
```python
cd ./MenuNER/ner
test.py 
--data_dir 
--pos2ix 
--model_path 
--max_length 
--test
--bert_checkpoint
--pos
--pos_embedding_dim
--char
--char_embedding_dim
--attention
```

## Usage
To easily test all the experiments aforementioned we also provide the following Google Colab notebook.
All of our experiments are run on Google Colab framework.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akatief/eurecom-evidence-generator/blob/develop/examples/TENET_colab.ipynb)

