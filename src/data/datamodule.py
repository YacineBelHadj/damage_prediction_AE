from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from pathlib import Path
import sqlite3
import pytorch_lightning as pl
import re
import torch
import numpy as np
import random

def set_global_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(123)


def collate_dict(batch):
    return {key: default_collate([d[key] for d in batch]) for key in batch[0]}

def execute_multiple_queries(cursor, query):
    """
    Executes multiple SQL statements in a single string by splitting them.
    """
    # Split the queries by semicolon and execute each
    statements = query.split(";")
    for statement in statements:
        statement = statement.strip()  # Remove whitespace
        if statement:  # Skip empty statements
            cursor.execute(statement) 
                
class PSDDataset(Dataset):
    def __init__(self, 
                 db_path: Path | str,
                 view_query: str,
                 view_table_name: str,
                 key_query: str,
                 key_name: str,
                 columns: list[str],
                 transform_func: list[callable],
                 cached: bool = False,
                 return_dict: bool = True):
        """
        Args:
            db_path: Path to the SQLite DB.
            view_query: A CREATE VIEW statement that defines a view in the database.
                        Example:
                          CREATE VIEW my_view AS SELECT id, feature1, feature2 FROM my_table
            key_query: A SELECT statement that returns all keys from the view.
                       Example:
                         SELECT id FROM my_view
            columns: The columns to fetch from the view for each sample, including the key column.
            transform_func: A list of transformations, one for each column in columns.
            cached: If True, prefetch all data into memory.
            return_dict: If True, return a dict {column_name: value} for each item.
        """
        self.db_path = db_path
        self.view_query = view_query
        self.key_query = key_query
        self.columns = columns
        self.transform_func = transform_func
        assert len(columns) == len(transform_func), "Columns and transform functions must have the same length."
        self.cached = cached
        self.return_dict = return_dict
        self.view_table_name = view_table_name

        # Connect to the database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.key_name = key_name

        # Create/Replace the view
        execute_multiple_queries(self.cursor, self.view_query)

        # Extract the view table name


        # Get all keys
        self.cursor.execute(self.key_query)
        self.keys = [row[0] for row in self.cursor.fetchall()]


        # Prepare query placeholders
        keys_placeholder = ', '.join(['?'] * len(self.keys))


        self.query_all_data = f"SELECT {', '.join(self.columns)} FROM {self.view_table_name} WHERE {self.key_name} IN ({keys_placeholder})"
        self.query_single_data = f"SELECT {', '.join(self.columns)} FROM {self.view_table_name} WHERE {self.key_name} = ?"

        # If cached, fetch all data now
        self.data = None
        if self.cached:
            self.data = self._fetch_all_data()


    def _apply_transform(self, data):
        # data is a tuple, so index by i
        return {col: self.transform_func[i](data[i]) for i, col in enumerate(self.columns)}

    def _fetch_all_data(self):
        """Fetch all data for all keys at once and store in a dictionary keyed by the primary key."""
        self.cursor.execute(self.query_all_data, self.keys)
        raw_data = self.cursor.fetchall()

        if not raw_data:
            raise ValueError("No data found for the specified keys in the view.")

        # raw_data rows look like: [(col1, col2, col3, ...), ...]
        # Identify the key column index
        return  [self._apply_transform(row) for row in raw_data]
    def _fetch_one_data(self, key):
        """
        Fetch data for a single key.
        """
        self.cursor.execute(self.query_single_data, (key,))
        raw_data = self.cursor.fetchone()

        if not raw_data:
            raise ValueError(f"No data found for key {key} in the view.")

 
        return self._apply_transform(raw_data)
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.keys):
            raise IndexError(f"Index {idx} out of range.")
        # key = self.keys[idx]  # We no longer need the key to index data if data is a list
        
        if self.cached:
            # Return preloaded data by index, not by key
            data = self.data[idx]
        else:
            # If not cached, we still need to fetch by key or by index. If you need to fetch by key:
            key = self.keys[idx]
            data = self._fetch_one_data(key)

        if not self.return_dict:
            return [data[col] for col in self.columns]
        return data

    def __del__(self):
        # Close DB connection when dataset is destroyed
        self.conn.close()

class PSDDataModule(pl.LightningDataModule):
    def __init__(self, 
                 db_path, 
                 view_query, 
                 view_table_name,  # Add this
                 key_query, 
                 key_name,         # Add this
                 columns, 
                 transform_func, 
                 batch_size=32, 
                 num_workers=4,
                 return_dict=False,
                 cached=False,
                 collate_function=None):
        super().__init__()
        self.db_path = db_path
        self.view_query = view_query
        self.view_table_name = view_table_name
        self.key_query = key_query
        self.key_name = key_name
        self.columns = columns
        self.transform_func = transform_func
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_dict = return_dict
        self.cached = cached
        self.is_setup = False
        self.collate_function = collate_function

    def setup(self, stage=None):
        if self.is_setup:
            print('Already set')
            return

        # Initialize the full dataset, now passing view_table_name and key_name
        self.dataset = PSDDataset(
            db_path=self.db_path,
            view_query=self.view_query,
            view_table_name=self.view_table_name,
            key_query=self.key_query,
            key_name=self.key_name,
            columns=self.columns,
            transform_func=self.transform_func,
            cached=self.cached,
            return_dict=self.return_dict
        )

        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        self.is_setup = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate_fn
        )

    def all_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        if self.collate_function:
            return self.collate_function(batch)
        if self.return_dict:
            return collate_dict(batch)
        else:
            return default_collate(batch)
    def __del__(self):
        if hasattr(self, 'dataset') and self.dataset is not None:
            del self.dataset
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            del self.train_dataset
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            del self.val_dataset
