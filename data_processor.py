import os
import pandas as pd
import numpy as np

def process_dataset(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8')
    chunks = []

    # Generate chunks for each column with dynamic insights
    for column in df.columns:
        col_data = df[column]
        if pd.api.types.is_numeric_dtype(col_data):
            stats = {
                'min': col_data.min() if not col_data.empty else None,
                'max': col_data.max() if not col_data.empty else None,
                'mean': col_data.mean() if not col_data.empty else None,
                'count': col_data.count(),
                'nulls': col_data.isnull().sum()
            }
            # Handle None values with proper formatting
            min_val = f"{stats['min']:.2f}" if stats['min'] is not None else "N/A"
            max_val = f"{stats['max']:.2f}" if stats['max'] is not None else "N/A"
            mean_val = f"{stats['mean']:.2f}" if stats['mean'] is not None else "N/A"
            chunk = f"Column {column}: Contains numeric data with minimum value {min_val}, maximum value {max_val}, average value {mean_val}, {stats['count']} non-null entries, and {stats['nulls']} nulls."
            chunks.append(chunk)
        else:
            unique_values = col_data.dropna().unique().tolist()
            sample_values = unique_values[:3] if unique_values else ['N/A']
            chunk = f"Column {column}: Contains non-numeric data with {len(unique_values)} unique values, including {', '.join(map(str, sample_values))} and more."
            chunks.append(chunk)

    # Generate group-by insights for any categorical column paired with numeric columns
    categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols:
            group_stats = df.groupby(cat_col).agg({col: ['mean', 'count'] for col in numeric_cols}).dropna()
            for _, row in group_stats.iterrows():
                group_key = row.name
                stats_str = ", ".join([f"{col}: average {row[(col, 'mean')]:.2f}, count {row[(col, 'count')]}"
                                    for col in numeric_cols])
                chunk = f"Group by {cat_col} '{group_key}': {stats_str}."
                chunks.append(chunk)

    return chunks