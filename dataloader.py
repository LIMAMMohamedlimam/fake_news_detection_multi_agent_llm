import pandas as pd

class dataset_loader:
    def __init__(self, dataset_path: str , mapper: dict):
        self.mapper = mapper
        for attr in mapper.values():
            setattr(self, attr, [])


        self._load_dataset(dataset_path , mapper)

    def __len__(self):
        attr = getattr(self , list(self.mapper.values())[0])
        return len(attr)

    def __getitem__(self, idx):
        return { attr.replace("s",""): getattr(self, attr)[idx] for attr in self.mapper.values() }
    
    def _load_dataset(self, dataset_path: str , mapper: dict):
        csv_data = pd.read_csv(dataset_path)
        length = len(csv_data)
        if length == 0:
            raise ValueError(f"Dataset at {dataset_path} is empty or not found.")
        for _,row in csv_data.iterrows():
            for col, attr in mapper.items():
                getattr(self, attr).append(row[col])
        

        # for _, row in csv_data.iterrows():
        #     self.texts.append(row['text'])
        #     self.titles.append(row['title'])
        #     self.subjects.append(row['subject'])
        #     self.labels.append(row['label'])

class fake_news_dataset:
    def __init__(self, dataset_path: str):
        self.titles : list[str] = []
        self.subjects : list[str] = []
        self.texts : list[str] = []
        self.labels : list[int] = [] # encode labels as integers 0 (fake), 1 (real)

        self._load_dataset(dataset_path)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'title': self.titles[idx],
            'text': self.texts[idx],
            'subject': self.subjects[idx],
            'label': self.labels[idx],
        }
    
    def _load_dataset(self, dataset_path: str):
        csv_data = pd.read_csv(dataset_path)
        length = len(csv_data)
        if length == 0:
            raise ValueError(f"Dataset at {dataset_path} is empty or not found.")

        for _, row in csv_data.iterrows():
            self.texts.append(row['text'])
            self.titles.append(row['title'])
            self.subjects.append(row['subject'])
            self.labels.append(row['label'])


if __name__ == "__main__":
    dataset = dataset_loader("data/fake_or_real_news.csv" , {
        'title': 'titles',
        'text': 'texts',
        'subject': 'subjects',
        'label': 'labels'
    })
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print("Sample data:", sample)

    gossipcop_dataset = dataset_loader("data/gossipcop_dataset.csv", mapper={
        'id': 'ids',
        'title': 'titles',
        'news_url': 'news_urls',
        'tweet_ids': 'tweet_ids',
        'label': 'labels',
    })
    print(f"GossipCop Dataset size: {len(gossipcop_dataset)}")
    sample_gc = gossipcop_dataset[0]
    print("GossipCop Sample data:", sample_gc)