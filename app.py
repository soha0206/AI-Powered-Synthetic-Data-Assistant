from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
import os
import pandas as pd
import numpy as np
import sqlite3
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from data_processor import process_dataset
from embedding_generator import generate_embeddings
import faiss
import pickle
import logging
from datetime import datetime, timedelta
from faker import Faker
from transformers import pipeline
import random
import uuid
import json
import re

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CLEANED_FOLDER = os.path.join(BASE_DIR, 'cleaned')
VECTOR_DB_FOLDER = os.path.join(BASE_DIR, 'vector_db')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLEANED_FOLDER'] = CLEANED_FOLDER
app.config['VECTOR_DB_FOLDER'] = VECTOR_DB_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# Configure Gemini API
API_KEY = "AIzaSyDa08tT7hax2OAc2iMMFQuinnaFRRO62ro"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
logger.debug("Gemini model configured successfully")

# Initialize SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load Hugging Face model locally
text_generator = pipeline("text-generation", model="distilgpt2")

# Initialize Faker
fake = Faker()

def profile_dataset(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')
    profiling_report = {}
    issues = []

    for column in df.columns:
        col_data = df[column]
        profiling_report[column] = {
            'inferred_type': str(col_data.dtype),
            'count': col_data.count(),
            'nulls': col_data.isnull().sum(),
            'unique_values': col_data.nunique(),
        }
        if np.issubdtype(col_data.dtype, np.number):
            profiling_report[column].update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'sum': col_data.sum(),
                'std_dev': col_data.std()
            })
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))].count()
            if outliers > 0:
                issues.append(f"Outliers detected in column {column}: {outliers} values.")
        else:
            profiling_report[column].update({
                'min': None,
                'max': None,
                'mean': None,
                'sum': None,
                'std_dev': None
            })

    if df.isnull().sum().sum() > 0:
        issues.append("Missing values detected in the dataset.")
    if df.duplicated().sum() > 0:
        issues.append(f"Duplicate rows detected: {df.duplicated().sum()} duplicates.")
    for column in df.columns:
        col_data = df[column]
        if col_data.dtype == 'object' and col_data.str.lower().nunique() < col_data.nunique():
            issues.append(f"Inconsistent casing in column {column}.")
        if 'date' in column.lower() and col_data.dtype == 'object':
            try:
                pd.to_datetime(col_data.dropna())
            except:
                issues.append(f"Invalid date formats in column {column}.")

    issues_text = " ".join(issues)
    if issues_text:
        prompt = f"Given the following data quality issues in a dataset: {issues_text} Suggest specific improvements to address these issues, including handling non-numeric values, missing values, duplicates, inconsistent casing, and invalid date formats."
        response = model.generate_content(f"Provide a concise suggestion based on: {issues_text}")
        suggestions = response.text if response else "No suggestions due to API error."
    else:
        suggestions = "No data quality issues detected. The dataset appears clean."

    return profiling_report, issues, suggestions

def clean_dataset(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')
    cleaned_df = df.copy()

    cleaned_df.columns = [col.strip().lower() for col in cleaned_df.columns]
    numeric_cols = [col for col in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[col])]
    for col in numeric_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        median_value = cleaned_df[col].median() if not cleaned_df[col].isna().all() else 0
        cleaned_df[col] = cleaned_df[col].fillna(median_value)
    non_numeric_cols = [col for col in cleaned_df.columns if not pd.api.types.is_numeric_dtype(cleaned_df[col])]
    for col in non_numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna('unknown')
        if 'date' in col:
            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce').dt.strftime('%Y-%m-%d')
    cleaned_df = cleaned_df.drop_duplicates()

    cleaned_filename = f"cleaned_{os.path.basename(filepath)}"
    cleaned_filepath = os.path.join(CLEANED_FOLDER, cleaned_filename)
    cleaned_df.to_csv(cleaned_filepath, index=False)

    # Generate vector database for all columns
    available_columns = [col.lower() for col in cleaned_df.columns.tolist()]
    logger.debug(f"Available columns for vectorization: {available_columns}")
    chunks = [cleaned_df.iloc[i:i + 100].to_string(index=False) for i in range(0, len(cleaned_df), 100)]
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    vector_store = {'chunks': chunks, 'index': index}
    vector_store_path = os.path.join(VECTOR_DB_FOLDER, f"{os.path.basename(cleaned_filename).replace('.csv', '')}_vector_store.pkl")
    with open(vector_store_path, 'wb') as f:
        pickle.dump(vector_store, f)
    logger.debug(f"Saved vector store to {vector_store_path} with {len(chunks)} chunks")

    return cleaned_filepath, cleaned_df.head().to_dict('records')
def generate_sql_schema(filepath):
    # Read the CSV file
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Prepare a summary of the dataset for AI analysis
    column_summaries = []
    total_rows = len(df)
    for column in df.columns:
        col_data = df[column]
        summary = {
            'name': column,
            'inferred_type': str(col_data.dtype),
            'non_null_count': int(col_data.count()),  # Convert to Python int
            'null_count': int(col_data.isnull().sum()),  # Convert to Python int
            'unique_values': int(col_data.nunique()),  # Convert to Python int
            'sample_values': col_data.dropna().head(5).tolist() if not col_data.empty else [],
            'is_numeric': np.issubdtype(col_data.dtype, np.number)
        }
        if summary['is_numeric']:
            summary.update({
                'min': float(col_data.min()) if pd.notna(col_data.min()) else None,
                'max': float(col_data.max()) if pd.notna(col_data.max()) else None,
                'mean': float(col_data.mean()) if pd.notna(col_data.mean()) else None
            })
        column_summaries.append(summary)

    # Create a prompt for the AI to generate a dynamic SQL schema
    prompt = f"""Given the following dataset summary, generate a dynamic SQL CREATE TABLE schema. 
    Use the following rules to infer SQL data types:
    - If inferred_type is 'int64' or sample_values contain only integers, use INTEGER.
    - If inferred_type is 'float64' or sample_values contain decimals, use REAL.
    - If 'date' is in the column name or sample_values match date patterns (e.g., YYYY-MM-DD), use DATE.
    - Otherwise, use TEXT.
    Suggest nullability (NOT NULL where non_null_count equals total rows: {total_rows}), 
    primary keys (unique column with no nulls and low unique_values relative to rows), 
    and UNIQUE constraints where unique_values approach total rows. 
    List columns first, followed by constraints (e.g., PRIMARY KEY, UNIQUE) on separate lines if needed.
    Do not include foreign keys or indexes unless explicitly suggested by the data.
    Return only the raw SQL code without any Markdown formatting (e.g., no ```sql or ```).

    Dataset Summary:
    {json.dumps(column_summaries, indent=2)}

    Return the schema as a string in SQL format, e.g.:
    CREATE TABLE table_name (
        column1 DATA_TYPE CONSTRAINTS,
        column2 DATA_TYPE CONSTRAINTS,
        ...
        [CONSTRAINT constraint_name PRIMARY KEY (column)]
    );"""

    # Generate schema using Gemini model
    response = model.generate_content(prompt)
    ai_generated_schema = response.text.strip() if response else None
    logger.debug(f"Raw AI-generated schema: {ai_generated_schema}")

    if not ai_generated_schema:
        # Fallback to basic schema generation if AI fails
        table_name = 'dataset'
        schema = [f"CREATE TABLE {table_name} ("]
        schema.append("    id INTEGER PRIMARY KEY AUTOINCREMENT,")

        dtype_mapping = {
            'object': 'TEXT',
            'int64': 'INTEGER',
            'float64': 'REAL',
            'datetime64[ns]': 'DATE'
        }

        quoted_columns = [f'"{col}"' for col in df.columns]
        for column in df.columns:
            dtype = str(df[column].dtype)
            sql_type = dtype_mapping.get(dtype, 'TEXT')
            if 'date' in column.lower():
                sql_type = 'DATE'
            null_constraint = ' NOT NULL' if df[column].count() == len(df) else ''
            schema.append(f'    "{column}" {sql_type}{null_constraint},')
        
        schema[-1] = schema[-1].rstrip(',')
        schema.append(");")
        ai_generated_schema = '\n'.join(schema)
        logger.warning("Fell back to basic schema generation due to AI failure.")
    else:
        # Clean Markdown syntax if present
        ai_generated_schema = re.sub(r'^```(?:sql)?\s*|\s*```$', '', ai_generated_schema, flags=re.MULTILINE)
        # Ensure the schema ends with a semicolon
        if not ai_generated_schema.endswith(';'):
            ai_generated_schema += ';'
        # Post-process to ensure valid schema
        lines = ai_generated_schema.split('\n')
        if len(lines) < 3 or not lines[1].strip():  # Minimum: CREATE, column, )
            logger.error(f"Invalid schema structure: {ai_generated_schema}")
            raise ValueError("AI-generated schema is malformed.")
        processed_lines = [lines[0]]  # Keep the CREATE TABLE line
        for line in lines[1:-1]:  # Process all lines except the last (closing parenthesis)
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2 or parts[0].upper() in ['PRIMARY', 'UNIQUE', 'CONSTRAINT']:  # Skip constraint lines
                processed_lines.append(line)
                continue
            column_name = parts[0].replace('"', '')
            if column_name not in df.columns:
                logger.warning(f"Skipping invalid column name: {column_name}")
                continue
            sql_type = parts[1]
            # Adjust types based on column name and data
            if 'age' in column_name.lower() or 'score' in column_name.lower():
                if any(str(val).replace('.', '').isdigit() for val in df[column_name].dropna().head()):
                    sql_type = 'INTEGER' if all(str(val).isdigit() for val in df[column_name].dropna().head()) else 'REAL'
            elif 'date' in column_name.lower():
                sql_type = 'DATE'
            null_constraint = ' NOT NULL' if df[column_name].count() == len(df) else ''
            processed_lines.append(f'    "{column_name}" {sql_type}{null_constraint},')
        # Ensure at least one column is present and add closing parenthesis
        if not any(line.strip().startswith('    "') for line in processed_lines[1:]):
            processed_lines.append('    dummy TEXT')
        processed_lines.append(');')
        ai_generated_schema = '\n'.join(processed_lines)
        logger.debug(f"Processed schema: {ai_generated_schema}")

    # Execute the schema and insert data
    db_path = os.path.join(BASE_DIR, 'dataset.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Extract table name from the schema
    table_name_match = re.search(r'CREATE TABLE (\w+)', ai_generated_schema)
    table_name = table_name_match.group(1) if table_name_match else 'dataset'
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.executescript(ai_generated_schema)

    # Prepare data for insertion
    quoted_columns = [f'"{col}"' for col in df.columns]
    placeholders = ','.join(['?' for _ in df.columns])
    columns = ','.join(quoted_columns)
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    logger.debug(f"Insert query: {insert_query}")

    # Handle date columns and convert data
    if any('date' in col.lower() for col in df.columns):
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d').fillna('1900-01-01')
    data = [tuple(row) for row in df.values]
    logger.debug(f"Sample data: {data[:1]}")

    try:
        cursor.executemany(insert_query, data)
        conn.commit()
    except sqlite3.OperationalError as e:
        logger.error(f"SQLite error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

    return ai_generated_schema
def query_dataset(query, filename, vector_db_folder='vector_db'):
    try:
        # Load the precomputed vector store
        vector_store_path = os.path.join(vector_db_folder, f"{filename}_vector_store.pkl")
        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {vector_store_path}. Please clean the dataset first.")

        with open(vector_store_path, 'rb') as f:
            vector_store = pickle.load(f)
        chunks, index = vector_store['chunks'], vector_store['index']
        logger.debug(f"Loaded vector store with {index.ntotal} vectors")

        # Generate query embedding
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        logger.debug(f"Query embedding shape: {query_embedding.shape}")

        # Perform similarity search
        k = min(5, index.ntotal)  # Dynamic k based on index size, capped at 5
        distances, indices = index.search(query_embedding, k)
        logger.debug(f"Search returned {k} nearest neighbors")

        # Select relevant context with all columns
        threshold = 0.8  # Stricter threshold for accuracy
        relevant_indices = [i for i, dist in zip(indices[0], distances[0]) if dist < threshold]
        if not relevant_indices:
            relevant_indices = indices[0][:min(k, index.ntotal)]
        
        context = [chunks[i] for i in relevant_indices if 0 <= i < len(chunks)]
        context_text = "\n".join(context) if context else "No relevant data found."
        logger.debug(f"Selected context length: {len(context_text)} characters")

        # Generate response based on all column data
        prompt = f"Answer the query '{query}' based *exclusively* on the following context, which includes all columns of the dataset. Do not invent data outside this context, and provide a precise response using exact values where available:\n\n{context_text}"
        response = model.generate_content(prompt)
        answer = response.text if response else "Oops! I couldn't generate a response due to an API error or insufficient context."
        logger.debug(f"Generated answer: {answer[:200]}...")

        return answer.strip(), context
    except Exception as e:
        logger.error(f"Error querying dataset: {e}")
        return f"Error processing query: {str(e)}", []

def infer_column_logic(columns):
    prompt = """Suggest Faker logic for the following columns:
"""
    for col in columns:
        prompt += f"- {col['name']}: {col['description']}\n"

    prompt += "Return Python Faker expressions in JSON list format: [{\"name\": ..., \"logic\": ...}, ...]"

    response = text_generator(prompt, max_new_tokens=100)[0]['generated_text']

    result = []
    for col in columns:
        logic = ""
        name = col["name"].lower()
        if "name" in name:
            logic = "fake.name()"
        elif "email" in name:
            logic = "fake.email()"
        elif "age" in name:
            logic = "random.randint(20, 60)"
        elif "date" in name or "dob" in name:
            logic = "fake.date_of_birth(minimum_age=20, maximum_age=60)"
        elif "address" in name:
            logic = "fake.address().replace('\\n', ', ')"
        elif "salary" in name or "income" in name:
            logic = "round(random.uniform(30000, 150000), 2)"
        elif "department" in name:
            logic = "random.choice(['HR', 'IT', 'Sales', 'Marketing', 'Finance'])"
        else:
            logic = "fake.word()"
        result.append({"name": col["name"], "logic": logic})

    return result

def generate_synthetic_dataset(dataset_name, columns_desc, num_rows):
    logic_map = infer_column_logic(columns_desc)
    data = []
    generated_values = {col['name']: set() for col in columns_desc}
    rows_generated = 0

    while rows_generated < num_rows:
        row = {}
        max_attempts = 10
        attempt = 0
        success = False
        while attempt < max_attempts and not success:
            is_unique = True
            for col in logic_map:
                name = col["name"]
                try:
                    value = eval(col["logic"])
                    if value in generated_values[name]:
                        is_unique = False
                        break
                    row[name] = value
                except Exception as e:
                    logger.warning(f"Error evaluating logic for {name}: {e}, using fallback")
                    row[name] = fake.word()
            
            if is_unique:
                for col in columns_desc:
                    generated_values[col['name']].add(row[col['name']])
                data.append(row)
                rows_generated += 1
                logger.debug(f"Generated row {rows_generated}/{num_rows}")
                success = True
            attempt += 1
        
        if not success and attempt == max_attempts:
            for col in logic_map:
                name = col["name"]
                if "name" in name.lower():
                    value = next((n for n in [fake.name() for _ in range(20)] if n not in generated_values[name]), f"Name_{attempt}")
                elif "age" in name.lower():
                    value = random.randint(20, 60)
                else:
                    value = fake.word()
                row[name] = value
            data.append(row)
            rows_generated += 1
            logger.debug(f"Generated row {rows_generated}/{num_rows} with fallback")

    df = pd.DataFrame(data)
    filename = f"{dataset_name.lower().replace(' ', '_')}_{num_rows}_rows.csv"
    filepath = os.path.join(CLEANED_FOLDER, filename)
    df.to_csv(filepath, index=False)
    return filepath, df.head().to_dict('records')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', message='No file uploaded', message_type='danger')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', message='No file selected', message_type='danger')
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            profiling_report, issues, suggestions = profile_dataset(filepath)
            cleaned_filepath, cleaned_preview = clean_dataset(filepath)
            return render_template('profile.html', filename=file.filename, report=profiling_report, issues=issues, suggestions=suggestions, message='File processed successfully', message_type='success')
        return render_template('upload.html', message='Invalid file format. Please upload a CSV.', message_type='danger')
    return render_template('upload.html')

@app.route('/profile/<filename>')
def profile(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    profiling_report, issues, suggestions = profile_dataset(filepath)
    return render_template('profile.html', filename=filename, report=profiling_report, issues=issues, suggestions=suggestions)

@app.route('/clean/<filename>')
def clean(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cleaned_filepath, cleaned_preview = clean_dataset(filepath)
    return render_template('cleaned.html', filename=filename, cleaned_filename=os.path.basename(cleaned_filepath), preview=cleaned_preview, message='Dataset cleaned successfully', message_type='success')

@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(CLEANED_FOLDER, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    return send_file(filepath, as_attachment=True)

@app.route('/download_schema/<filename>')
def download_schema(filename):
    filepath = os.path.join(CLEANED_FOLDER, filename)
    sql_schema = generate_sql_schema(filepath)
    schema_path = os.path.join(CLEANED_FOLDER, f"{os.path.splitext(filename)[0]}_schema.sql")
    with open(schema_path, 'w') as f:
        f.write(sql_schema)
    return send_file(schema_path, as_attachment=True, download_name=f"{os.path.splitext(filename)[0]}_schema.sql")

@app.route('/query/<filename>', methods=['GET', 'POST'])
def query(filename):
    if request.method == 'POST':
        query_text = request.form.get('query')
        if query_text:
            try:
                answer, context = query_dataset(query_text, os.path.basename(filename).replace('.csv', ''), VECTOR_DB_FOLDER)
                return render_template('query.html', filename=filename, query=query_text, answer=answer, context=context, message='Query processed successfully', message_type='success')
            except Exception as e:
                return render_template('query.html', filename=filename, query=query_text, answer=f"Error processing query: {str(e)}", context=[], message_type='danger')
    return render_template('query.html', filename=filename, query='', answer='', context=[])

@app.route('/generate', methods=['GET', 'POST'])
def generate_dataset():
    if request.method == 'POST':
        input_str = request.form.get('input_str', '').strip()
        num_rows = int(request.form.get('num_rows', 100))
        if not input_str:
            return render_template('generate.html', message='Input is required', message_type='danger')
        
        # Parse input: dataset name followed by column definitions
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) < 1:
            return render_template('generate.html', message='Invalid input format', message_type='danger')
        
        dataset_name = parts[0]
        columns_desc = []
        for part in parts[1:]:
            if ' - ' in part:
                name, desc = part.split(' - ', 1)
                columns_desc.append({'name': name.strip(), 'description': desc.strip()})
            else:
                columns_desc.append({'name': part.strip(), 'description': 'Generic value'})
        
        if not columns_desc:
            return render_template('generate.html', message='At least one column is required', message_type='danger')
        
        logger.debug(f"Parsed input: dataset_name={dataset_name}, num_rows={num_rows}, columns_desc={columns_desc}")
        try:
            generated_filepath, preview = generate_synthetic_dataset(dataset_name, columns_desc, num_rows)
            logger.debug(f"Generated dataset: {generated_filepath} with {num_rows} rows")
            return render_template('generated.html', filename=os.path.basename(generated_filepath), preview=preview, message=f'Generated {num_rows} rows successfully', message_type='success', num_rows=num_rows)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return render_template('generate.html', message=f'An error occurred during generation: {str(e)}', message_type='danger')

    return render_template('generate.html')

@app.route('/confirm_generate', methods=['POST'])
def confirm_generate():
    # This route is now redundant since generation is handled in /generate
    return redirect(url_for('generate_dataset'))

if __name__ == '__main__':
    app.run(debug=True)
