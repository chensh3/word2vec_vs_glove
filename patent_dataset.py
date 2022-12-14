from datasets import load_dataset
from pprint import pprint
import pandas as pd
pd.options.mode.chained_assignment = None

dataset_dict = load_dataset('HUPD/hupd',
                            name='sample',
                            data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather",
                            icpr_label=None,
                            train_filing_start_date='2016-01-01',
                            train_filing_end_date='2016-01-21',
                            val_filing_start_date='2016-01-22',
                            val_filing_end_date='2016-01-23',
                            )

print('Loading is done!')

# Print dataset dictionary contents and cache directory
print('Dataset dictionary contents:')
pprint(dataset_dict['train'])


# Print info about the sizes of the train and validation sets
print(f'Train dataset size: {dataset_dict["train"].shape} \n')

train = dataset_dict["train"]
df_full = pd.DataFrame(train)
df = df_full[["patent_number", "title", 'abstract', 'background', 'summary', 'description', ]]
cols=['abstract','background','summary','description']
df.loc[:,"full_text"] = df[cols].apply(lambda row: ' \n '.join(row.values.astype(str)), axis=1)

