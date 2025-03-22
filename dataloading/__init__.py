from datasets import load_dataset


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
