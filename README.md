# English - Marathi Language Translation 

* Build EN-MR and MR-EN language translation models then merged them into a bidirectional translation model by extracting the encoder and decoder components from each model and creating a new EncoderDecoderModel using the HuggingFace transformers library.
* Leveraging Huggingface's pretrained multilingual translation models, I developed a English-to-Marathi and Marathi to English translation model by fine-tuning hyperparameters and utilizing AutoModelForSeq2seqLM, AutoTokenizer, AutoConfig, Se
* Compared three different models i.e.
    * [Mbart](https://huggingface.co/facebook/mbart-large-50)
    * [AI4Bharat](https://huggingface.co/ai4bharat/indic-bert)
    * [Helsinki-NLP](https://huggingface.co/Helsinki-NLP)
* Achieved remarkable results with the Helsinki-NLP which gave a loss rate of 0.5174, surpassing the other models.

## Data

The dataset I have collected is a comprehensive collection of English and Marathi translations obtained from various publicly available resources. It contains a total of 3,517,283 rows, making it a substantial dataset for language translation tasks. The dataset size is approximately 451 MB, indicating the richness of the data contained within it.
To access and utilize this dataset conveniently, it can be downloaded and loaded using the Hugging Face datasets library.

```
from datasets import load_dataset
dataset = load_dataset("anujsahani01/English-Marathi")
```

## My Models:

The models can be accessed and tested on Huggingface, using the below links.
* [Mbart[EN-MR]](https://huggingface.co/anujsahani01/finetuned_mbart) , [Mbart[MR-EN]](https://huggingface.co/anujsahani01/finetuned_Mbart_mr_en)
* [Helsinki-NLP[EN-MR]](https://huggingface.co/anujsahani01/finetuned_Helsinki-NLP-en-mr) , [Helsinki-NLP[MR-EN]](https://huggingface.co/anujsahani01/finetuned_Helsinki-NLP-mr-en)
* [AI4Bharat[EN-MR]](https://huggingface.co/anujsahani01/finetuned_AI4Bharat_en_mr) , [AI4Bharat[MR-EN]](https://huggingface.co/anujsahani01/finetuned_AI4Bharat_mr_en)


## Hyperparameter(s) that best suited the model.

After series of experiments and trials i finally found this set of Hyperparameters on which my model performed best.

| Helsinki-NLP                          |     Mbart                         |   AI4Bharat   |
| ----------------------------          | -------------                     | ------------- |
| learning rate : 0.0005                | learning rate : 0.0005            | learning rate : 0.0005
| max_steps : 10000                     | max_steps : 10000 | max_steps : 8000
| warmup steps : 50                     | warmup steps : 50 | warmup steps : 50
| weight_decay : 0.01                   | weight_decay : 0.01 | weight_decay : 0.01
| per_device_train_batch_size : 64      | per_device_train_batch_size : 12 | per_device_train_batch_size : 12
| per_device_eval_batch_size : 64       | per_device_eval_batch_size : 12 | per_device_eval_batch_size : 12
| evaluation_strategy : ‘no’            | evaluation_strategy : ‘no’ | evaluation_strategy : ‘no’ 
| num_train_epochs : 1                  | num_train_epochs : 1 | num_train_epochs : 1
| remove_unused_columns : False         | remove_unused_columns : False     | remove_unused_columns : False
      

## Results:

The following losses were obtained for <ins>English to Marathi</ins> Language Translation Model.
The best results were obtained using a fine-tuned Helsinki-NLP model.

| Helsinki-NLP  |     Mbart     |   AI4Bharat   |
| ------------- | ------------- | ------------- |
|    0.5174     |    0.8225     |    0.9779     |


The following losses were obtained for <ins>Marathi to English</ins> Language Translation Model.
The best results were obtained using a fine-tuned Mbart model.

| Helsinki-NLP  |     Mbart     |   AI4Bharat   |
| ------------- | ------------- | ------------- |
|    0.6818     |    0.6712     |    0.7775     |


## Feedback

If you have any feedback, please reach out to me at: [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/anuj-sahani-34363725b) 

Author: [@anujsahani01](https://github.com/anujsahani01)
