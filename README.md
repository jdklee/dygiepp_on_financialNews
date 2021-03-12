# dygiepp_on_financialNews
Using pretrained dyGIE++ model, evaluates labelled financial news articles.

## Dygie++ paper
https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8

## Origianl repo:
https://github.com/dwadden/dygiepp


## Requirements:
** MAKE SURE THAT YOU HAVE THE DYGIE++.IPYNB OPEN WHEN GOING THROUGH THE README.

1. Clone the current repository with data and notebooks from https://github.com/jdklee/dygiepp_on_financialNews
2. create virtual environment and install requirements.txt by executing:

```
conda create --name dygiepp python=3.7

pip install -r requirements.txt

conda develop .   # Adds DyGIE to your PYTHONPATH
```

3. Download pretrained model from https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/ace05-event.tar.gz
--> Since our dataset consists of financial news articles, the model trained on ace05 dataset is the most appropriate. 


## Project explanation
For the scope of our project, we only deal with NER (Named Entity Tagging) task. 
Due to constraints in labelling the news article, currently we only have 59 news articles with Named Entity Tags.

We evaluate the dataset, collecting three metrics: precision, recall, and f-1 score


## Dataset description
The full dataset consists of 29,567 financial news. Articles are grouped by company name and sorted by publication time. We get rid of the title and only use the text body in the articles.

## Data Pre-processing
In the case where your unlabeled data are stored as a directory of .txt files (one file per document), you can run python scripts/data/new-dataset/format_new_dataset.py [input-directory] [output-file] to format the documents into a jsonl file, with one line per document. If your dataset is scientific text, add the --use-scispacy flag to have SciSpacy do the tokenization.

The code below takes in a dataFrame object and generates .txt file per row. 

```python
def format_data(df):
    firms=df.company.unique()
    for firm in firms:
        temp=df[df.company==firm].reset_index(drop=True)
        for i in range(len(temp)):
            print(firm,i)
            txt=open("/Users/jdklee/Documents/StatisticalML/articles/{}.txt".format(firm+"_"+str(i)),"w")
            txt.write(temp.text[i])
            txt.close()
```

Once the data is partitioned into separate .txt files, we can run the following script:

```
scripts/data/new-dataset/format_new_dataset.py [input-directory] [output-file]
```

Usage example:
```
python /Users/jdklee/Documents/dygiepp/scripts/new-dataset/format_new_dataset.py /Users/jdklee/Documents/StatisticalML/articles /Users/jdklee/Documents/StatisticalML/realArticlesACE05.jsonl ace05
```


## Predictions
To speed up the label process, we first make predictions on the unlabelled dataset using pretrained dyGIE++. Once preprocessing is done, we can run the following on terminal:

```
allennlp predict pretrained/[name-of-pretrained-model].tar.gz \
    [input-path] \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file [output-path] \
    --cuda-device [cuda-device]
```

Usage example:
```
allennlp predict /Users/jdklee/Documents/dygiepp/ace05-event.tar.gz \
    /Users/jdklee/Documents/StatisticalML/aceEvent.jsonl \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file /Users/jdklee/Documents/StatisticalML/aceEventPredictionResults \
    --cuda-device -1

```


## Hand label
Currently, we have 59 news articles with NER labels. 

To speed up the label process, we first make predictions. Then, we lay out the sentence-span and predicted labels to start generating ground truth labels, sentence by sentence in each article. Then, we must create a NER object using the labels and add these to the dataset (dictionary).

```python

def generate_NER(lst,sent):
    ret=[document.NER(ner=tup, sentence=sent, sentence_offsets=True) for tup in lst]
    #print(ret)
    return ret
    
def add_NER_from_dict(dicto):
    for idx,i in enumerate(dicto):
        sent=i[0]
        nerList=i[1]
        try:
            sent.ner=generate_NER(nerList, sent)
        except:
            print(idx)
            print(nerList)
            print("------")
            
def get_labelled_docs(dataset):
    documents=[]
    for doc in dataset:
        for sent in doc:
            if sent.ner:
                documents.append(doc)
                break
    return documents
    
def handle_empty_ner(documents):
    for idx_doc, doc in enumerate(documents):
        for idx_sent, sent in enumerate(doc):
            if not sent.ner:
                sent.ner=[]
```

Usage example is available on labellingData.pynb

The labels used are:
```
"WEA" = weapon
"PER" = person
"VEH" = vehicle
"GPE" = geopolitical entity
"LOC" = location
"ORG" = organization
"FAC" = facility
```

## Evaluating

Now that we have NER labels, we can evaluate the result of predictions.

The code below sets up the dataset for evaluation.
```
#Create dataset that only contains labelled articles using filtered documents, extracted from get_labelled_docs(dataset)

labelled_test=document.Dataset(documents)
labelled_test.to_jsonl("/Users/jdklee/Documents/StatisticalML/labelled_test_data_events.jsonl")
```

Now, we can evaluate the model by executing:

```
allennlp evaluate \
  [model-file] \
  [data-path] \
  --cuda-device [cuda-device] \
  --include-package dygie \
  --output-file [output-file] # Optional; if not given, prints metrics to console.
```

Usage example:
```
allennlp evaluate \
  /Users/jdklee/Downloads/ace05-event.tar.gz \
  /Users/jdklee/Documents/StatisticalML/labelled_test_data_events.jsonl \
  --include-package dygie \
  --cuda-device -1  \
```

** We need to ensure to use older versions of allennlp to not break the model. This is modified from the original requirements.txt in dygiepp repo. Make sure you clone our repo to ensure the model runs.

Below is the requirements.txt file for no confusion on this. This is crucial for the notebook to run smoothly.
```
# Usage: `pip install -r requirements.txt``

# Modeling
allennlp==1.1.0
allennlp_models==1.0

# Data munging
pandas==0.25.2
beautifulsoup4==4.8.1
lxml
python-Levenshtein

# Hyperparameter search
optuna>=2.1.0
```

## Results

The following shows that although the pretrained model is okay in Named Entity Tag task on unseen dataset, it would be best if more labels could be generated and the model is finetuned on the data.

```
  "_ace-event__ner_precision": 0.6250901225666907,
  "_ace-event__ner_recall": 0.6861891571032845,
  "_ace-event__ner_f1": 0.6542161856253537,
```
