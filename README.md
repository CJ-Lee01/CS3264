# CS3264

## Project Structure
- `dataloading` is for code for loading data. If data needs to be downloaded, provide instructions on how to download and place the data. The code can be stored in `__init__.py`.
  - When a new dataset is added, ideally add it to the list of currently used datasets. The minimum expectation is to add the source for the dataset within the code and if there are any unusual transformations, document it.
- `model_training` is for code for training a model. Since the intricacies of training a model can be different, each training code can be in its own file.

## Currently used datasets
Unless for a specific reason, the implementation of providing a dataset should provide the following
- Pandas Dataframe, with column of the order: Text, Label
- Train, Test, Eval(optional) split where possible.

1. Hasib18 fake news dataset
   - src: https://huggingface.co/datasets/Hasib18/fake-news-dataset
   - function: dataloading.get_hasib18_fns
   - keyword argument: include_instruction(default false)
  
2. FineFake : A Knowledge-Enriched Dataset for Fine-Grained Multi-Domain Fake News Detection
   - src: https://github.com/Accuser907/FineFake?tab=readme-ov-file
   - function: dataloading.get_multilingual_dataset
  
3. MCFEND: A Multi-source Benchmark Dataset for Chinese Fake News Detection
   - src: https://trustworthycomp.github.io/mcfend/
   - function: dataloading.get_multilingual_dataset

4. Kaggle fake-and-real-news-dataset
   - src: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
   - function: dataloading.get_multilingual_dataset

## Dependencies
### Datasets
- [Datasets](https://huggingface.co/docs/hub/datasets-usage)
- Pandas

### Evaluation
- sci-kit learn (not added yet)
- [torcheval](https://pytorch.org/torcheval/main/torcheval.metrics.html) (not added yet)

### Models (not added yet)
- sci-kit learn 
- Pytorch
- Huggingface: To be specified

