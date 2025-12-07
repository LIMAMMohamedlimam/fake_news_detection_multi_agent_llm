
import pandas as pd
import kagglehub
import os

fake_news_path = 'data/news_dataset.csv'
real_news_path = 'data/real_news_dataset.csv'



def stack_datasets(fake_path: str = fake_news_path, real_path: str = real_news_path) :
    fake_data = pd.read_csv(fake_path)
    fake_data['label'] = 0  # Fake label
    real_data = pd.read_csv(real_path)
    real_data['label'] = 1  # Real label
    
    combined_data = pd.concat([fake_data, real_data], ignore_index=True)
    combined_data.to_csv('data/FTDS.csv', index=False)  # Backup before shuffling
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataset

    return combined_data



if __name__ == "__main__":

    # Download latest version
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

    print("Path to dataset files:", path)
    stack_datasets(fake_path=os.path.join(path, 'Fake.csv'),
                   real_path=os.path.join(path, 'True.csv'))
    df = pd.read_csv('data/fakeTrueDS.csv')
    print(df.head())    
    print(f"Dataset shape: {df.shape}")
    print(df['label'].value_counts())
    print(df["subject"].value_counts())