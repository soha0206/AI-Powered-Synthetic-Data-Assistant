import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_embeddings(filepath, filename, vector_db_folder, available_columns, 
                       custom_group_cols=None, agg_methods=['mean', 'sum'], 
                       min_rows=1, max_rows=None, embedding_model='all-MiniLM-L6-v2', 
                       batch_size=32, normalize=True, chunk_size=100, callback=None):
    """
    Generate embeddings for the dataset with dynamic column handling and configurable options.
    
    Args:
        filepath (str): Path to the cleaned CSV file
        filename (str): Base filename without extension
        vector_db_folder (str): Folder to save vector database
        available_columns (list): List of column names present in the DataFrame
        custom_group_cols (list, optional): Custom columns to group by
        agg_methods (list, optional): Aggregation methods (e.g., ['mean', 'sum'])
        min_rows (int, optional): Minimum number of rows to process
        max_rows (int, optional): Maximum number of rows to process
        embedding_model (str, optional): Model for embeddings
        batch_size (int, optional): Batch size for embedding generation
        normalize (bool, optional): Whether to normalize embeddings
        chunk_size (int, optional): Size of text chunks for embedding
        callback (function, optional): Callback function for progress
    """
    # Load the DataFrame
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Ensure minimum rows
    if len(df) < min_rows:
        logger.warning(f"Dataset has {len(df)} rows, less than min_rows {min_rows}. Using all available rows.")
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    
    # Copy and filter columns
    dept_stats = df.copy()
    valid_columns = [col for col in available_columns if col in df.columns]
    logger.debug(f"Valid columns after filtering: {valid_columns}")
    
    # Grouping and aggregation
    group_by_cols = custom_group_cols if custom_group_cols else \
        [col for col in valid_columns if col.lower() in ['department', 'category', 'group']]
    
    if group_by_cols:
        agg_cols = [col for col in valid_columns if pd.api.types.is_numeric_dtype(df[col]) and col not in group_by_cols]
        agg_dict = {col: agg_methods for col in agg_cols}
        if agg_dict:
            dept_stats = df.groupby(group_by_cols).agg(agg_dict).reset_index()
    else:
        logger.warning("No suitable group column found, using raw data.")
    
    # Prepare text data
    text_data = dept_stats.astype(str).agg(' '.join, axis=1).tolist()
    
    # Load the SentenceTransformer model (fresh every time)
    embedder_instance = SentenceTransformer(embedding_model)
    embeddings = embedder_instance.encode(
        text_data,
        batch_size=batch_size,
        normalize_to_unit=normalize,
        convert_to_numpy=True
    )
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save vector store
    vector_store = {'chunks': text_data, 'index': index}
    vector_store_path = os.path.join(vector_db_folder, f"{filename}_vector_store.pkl")
    with open(vector_store_path, 'wb') as f:
        pickle.dump(vector_store, f)
    
    # Callback if needed
    if callback:
        callback(vector_store_path)
    
    logger.debug(f"Embeddings generated and saved to {vector_store_path} with {len(text_data)} chunks")
