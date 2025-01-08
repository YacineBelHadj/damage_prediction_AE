from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pytorch_lightning as pl

class PSDDataset(Dataset):
    def __init__(self, dataframe, feature_columns:list[str], transform:list[callable]):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the data.
            feature_columns (list of str): Columns with PSD data.
            transform (list of callables, optional): List of transformations for each feature column.
        """
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.transform =  transform
        assert len(self.feature_columns) == len(self.transform), (
            "The number of feature columns and transformations must be equal"
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        features = dict()
        for i, col in enumerate(self.feature_columns):
            features[col]= self.transform[i](row[col])
                        
        return features

class PSDDataModule(pl.LightningDataModule):
    def __init__(self, dataframe, feature_columns,transform, batch_size=256, split_ratio=0.1):
        """
        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            feature_columns (list of str): List of column names containing PSD data.
            target_column (str): Column name for the target labels.
            batch_size (int): Batch size for the DataLoader.
            split_ratio (float): Ratio for splitting data into validation set from the training data.
        """
        super().__init__()
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.transform = transform
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        """
        Prepare the data splits for train and validation.
        """

        # Define the dataset
        dataset = PSDDataset(
            dataframe=self.dataframe,
            feature_columns=self.feature_columns,
            transform=self.transform
        )

        # Compute split sizes
        total_size = len(dataset)
        val_size = int(total_size * self.split_ratio)
        train_size = total_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

