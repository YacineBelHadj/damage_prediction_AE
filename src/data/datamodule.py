from torch.utils.data import Dataset
from pathlib import Path
import sqlite3
from torch.utils.data import DataLoader, random_split, default_collate
import pytorch_lightning as pl

def collate_dict(batch):
    return {key: default_collate([d[key] for d in batch]) for key in batch[0]}

class PSDDataset(Dataset):
    def __init__(self, db_path: Path | str,
                 key_query: str,
                 data_query: str,
                 key_name: str,
                 transform_func: list[callable],
                 cached: bool = False,
                 return_dict: bool = False):
        self.db_path = db_path
        self.key_query = key_query  # Query template with placeholders
        self.data_query = data_query  # Column used to fetch keys
        self.transform_func = transform_func
        self.cached = cached
        self.return_dict = return_dict

        # Connect to the database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Format the query to fetch all keys (distinct values of index_column_name)

        self.keys = [row[0] for row in self.cursor.execute(key_query).fetchall()]
    
        # Generate placeholders for keys
        keys_placeholder = ", ".join("?" * len(self.keys))

        # Correctly construct `self.query_all_data`
        self.query_all_data =  self.data_query + f" WHERE {key_name} IN ({keys_placeholder})"

    def _apply_transform(self, data):
        """
        Apply transformations to the data.
        """
        return [func(value) for func, value in zip(self.transform_func, data)]

    def _fetch_all_data(self):
        """
        Fetch all data for the specified keys.
        """
        self.cursor.execute(self.query_all_data, self.keys)
        raw_data = self.cursor.fetchall()

        if not raw_data:
            raise ValueError("No data found for the specified keys.")

        # Create a dictionary mapping keys to transformed data
        data_dict = {}
        for row in raw_data:
            key = row[0]  # Assuming the first column is the key
            data_dict[key] = self._apply_transform(row[1:])  # Transform the remaining columns
        return data_dict

    def _fetch_one_data(self, key):
        """
        Fetch data for a single key.
        """
        self.cursor.execute(self.query_one_data, (key,))
        raw_data = self.cursor.fetchone()

        if not raw_data:
            raise ValueError(f"No data found for key: {key}")

        return self._apply_transform(raw_data[1:])  # Skip the key column

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Fetch a single item by index.
        """
        key = self.keys[idx]

        if self.cached:
            # Return preloaded data
            data = self.data[key]
        else:
            # Fetch data dynamically
            data = self._fetch_one_data(key)

        if self.return_dict:
            # Convert to dictionary format if required
            return {col: data[i] for i, col in enumerate(self.data_columns)}
        return data

    def __del__(self):
        self.conn.close()


class PSDDataModule(pl.LightningDataModule):
    def __init__(self, db_path, query_key, columns, transform_func, batch_size=32, num_workers=4,return_dict=False,cached=False,collate_function=None):
        super().__init__()
        self.db_path = db_path
        self.query_key = query_key
        self.columns = columns
        self.transform_func = transform_func
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_dict = return_dict  
        self.cached = cached
        self.is_setup = False
        self.collate_function = collate_function
        
    def setup(self,stage=None):
        if self.is_setup:
            print('Already set')
        else:
            self._setup(stage)
            self.is_setup=True
    def _setup(self, stage=None):
        # Initialize the full dataset
        self.dataset = PSDDataset(
            db_path=self.db_path,
            query_key=self.query_key,
            columns=self.columns,
            transform_func=self.transform_func,
            cached=self.cached,
            return_dict=self.return_dict
            
        )

        # Split the dataset into train, val, and test sets
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        # Use random_split to split the dataset
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self.collate_fn
        )
    def all_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self.collate_fn
        )
        
    def collate_fn(self, batch):
        if self.collate_function:
            return self.collate_function(batch)  # Call the provided function on the batch
        if self.return_dict:
            return collate_dict(batch)
        else:
            return default_collate(batch)
