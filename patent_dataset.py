from datasets import load_dataset
# Pretty print
from pprint import pprint
import pandas as pd

dataset_dict = load_dataset('HUPD/hupd',
                            name='sample',
                            data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather",
                            icpr_label=None,
                            train_filing_start_date='2016-01-01',
                            train_filing_end_date='2016-01-21',
                            val_filing_start_date='2016-01-22',
                            val_filing_end_date='2016-01-31',
                            )

print('Loading is done!')

# Dataset info
print(dataset_dict)

# Print dataset dictionary contents and cache directory
print('Dataset dictionary contents:')
pprint(dataset_dict)
print('Dataset dictionary cached to:')
pprint(dataset_dict.cache_files)

# Print info about the sizes of the train and validation sets
print(f'Train dataset size: {dataset_dict["train"].shape}')
print(f'Validation dataset size: {dataset_dict["validation"].shape}')

train = dataset_dict["train"]
df = pd.DataFrame(train)
# data = df["background"][1]






