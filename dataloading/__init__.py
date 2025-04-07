from datasets import load_dataset
import pandas as pd
import pickle
import os

def get_hasib18_fns(*, include_instruction=False):
    # https://huggingface.co/datasets/Hasib18/fake-news-dataset
    # Text came with instructions, likely for ChatGPT to test if a model can classify fakenews.
    prefix = "Instruction: Classify the following news article as real or fake.\n\nInput: "
    suffix = "\n\nOutput: fake"
    l_pre = len(prefix);
    l_suf = len(suffix)

    ds = load_dataset("Hasib18/fake-news-dataset")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    if not include_instruction:
        train_df["text"] = train_df["text"].apply(lambda x: x[l_pre:-l_suf])
        test_df["text"] = test_df["text"].apply(lambda x: x[l_pre:-l_suf])

    return train_df, test_df

def get_multilingual_dataset():

    #fake and real news from Sina Weibo [2] ranging from December 2014 to March 2021.
    #assume no data leakage between train and test
    '''
    @INPROCEEDINGS{mcfend,
      title={MCFEND: A Multi-source Benchmark Dataset for Chinese Fake News Detection}, 
      author={Li, Yupeng and He, Haorui and Bai, Jin and Wen, Dacheng},
      booktitle={Proc.~of WWW}, 
      year={2024},
    }
    download data here
    https://trustworthycomp.github.io/mcfend/
    '''

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    english_train_true = pd.read_csv(os.path.join(data_dir, 'True.csv')).sample(6000)
    english_train_true['label'] = 0
    english_train_fake = pd.read_csv(os.path.join(data_dir, 'Fake.csv')).sample(6000)
    english_train_fake['label'] = 1
    
    #merge the two dataframes
    english_train = pd.concat([english_train_true, english_train_fake], ignore_index=True)
    english_train = english_train[english_train['text'].notna()]
    english_train = english_train[['text', 'label']]
    
    #balance the test dataset
    modern_dataset = pd.read_csv(os.path.join(data_dir, 'FineFake.csv'))
    #split the dataset into half
    english_test = modern_dataset.sample(frac=0.5, random_state=42).copy()
    #put remianing data into val
    english_val = modern_dataset.drop(english_test.index)
    print("english test set size: ", len(english_test))
    print("english val set size: ", len(english_val))

    english_test = english_test[['text', 'label']]
    english_val = english_val[['text', 'label']]

    chinese_news_df = pd.read_csv(os.path.join(data_dir, 'mcfend.csv'))
    #map from text to int labels
    chinese_news_df['label'] = chinese_news_df['label'].apply(lambda x: 0 if x == '事实' else 1)
    cleaned_df = chinese_news_df[chinese_news_df['content'].notna() & chinese_news_df['publish_time'].notna()]
    train_end_date = '2021-02-28'
    val_end_date = '2022-02-28'
    chinese_train = cleaned_df[cleaned_df['publish_time'] < train_end_date].copy()
    chinese_val = cleaned_df[(cleaned_df['publish_time'] >= train_end_date) & (cleaned_df['publish_time'] < val_end_date)].copy()
    chinese_test = cleaned_df[cleaned_df['publish_time'] >= val_end_date].copy()


    #rename columns
    chinese_test.rename(columns={'content':'text'}, inplace=True)
    chinese_train.rename(columns={'content':'text'}, inplace=True)
    chinese_val.rename(columns={'content':'text'}, inplace=True)

    #retain only text and label columns
    chinese_test = chinese_test[['text', 'label']]
    chinese_train = chinese_train[['text', 'label']]
    chinese_val = chinese_val[['text', 'label']]

    merged_train = pd.concat([english_train, chinese_train], ignore_index=True)
    merged_test = pd.concat([english_test, chinese_test], ignore_index=True)
    merged_val = pd.concat([english_val, chinese_val], ignore_index=True)

    #shuffle the dataframes
    merged_train = merged_train.sample(frac=1).reset_index(drop=True)
    merged_test = merged_test.sample(frac=1).reset_index(drop=True)
    merged_val = merged_val.sample(frac=1).reset_index(drop=True)

    print("train set size: ", len(merged_train))
    print("test set size: ", len(merged_test))
    print("val set size: ", len(merged_val))

    return merged_train, merged_test, merged_val
