import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re
from time import time
import gc
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
SAMPLE_FRACTION = 0.5

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return ' '.join(text.split())
    return ''

def create_features(df, is_training=True):
    result_df = pd.DataFrame(index=df.index)
    
    print("Creating features...")
    #.loc was something I have to look up to use, this is used for indexing 
    result_df.loc[:, 'Id'] = df['Id'].copy() # making a copy of the file
    # text cleaning
    result_df.loc[:, 'clean_text'] = df['Text'].fillna('').apply(clean_text)
    result_df.loc[:, 'clean_summary'] = df['Summary'].fillna('').apply(clean_text)
    
    # Numerical feature
    result_df.loc[:, 'helpfulness_ratio'] = (
        df['HelpfulnessNumerator'].fillna(0) / 
        df['HelpfulnessDenominator'].fillna(1).replace(0, 1)
    )
    
    # Length features
    result_df.loc[:, 'text_length'] = df['Text'].fillna('').str.len()
    result_df.loc[:, 'summary_length'] = df['Summary'].fillna('').str.len()
    
    #  Score column only for training data
    if is_training and 'Score' in df.columns:
        result_df.loc[:, 'Score'] = df['Score']
    
    return result_df

def batch_process(df, batch_size=50000, is_training=True):
    # This function is added for effciency reasons. Processing data in batches helped to speed up the process
    all_features = []
    
    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx].copy()
        batch_features = create_features(batch, is_training)
        all_features.append(batch_features)
        gc.collect()
    
    return pd.concat(all_features, axis=0)

def main():
    start_time = time() # I was intersted in tracking the time, though it might not be as accurate as I want 
    
    # Load data
    dtypes = {
        'Id': 'int32',
        'ProductId': 'string',
        'UserId': 'string',
        'HelpfulnessNumerator': 'float32',
        'HelpfulnessDenominator': 'float32',
        'Score': 'float32',
        'Time': 'int32' # these are modified to 32 also for effciency reasons
    }
    
    complete_train_df = pd.read_csv("./data/train.csv", dtype=dtypes)
    test_df = pd.read_csv("./data/test.csv")
    
    print("\nPreparing test data")
    test_reviews = pd.merge(
        test_df[['Id']], 
        complete_train_df[['Id', 'Text', 'Summary', 'HelpfulnessNumerator', 'HelpfulnessDenominator']],
        on='Id',
        how='left'
    )
    
    # Prepare training data
    print("\nPreparing training data")
    train_df = complete_train_df.dropna(subset=['Score'])
    train_df = train_df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_SEED)
 
    del complete_train_df
    gc.collect() # added for effciency 
    
    # Process data
    print("\nProcessing training data")
    train_features = batch_process(train_df, is_training=True)
    
    print("\nProcessing test data")
    test_features = batch_process(test_reviews, is_training=False)
    
    # Define features
    numerical_features = ['helpfulness_ratio', 'text_length', 'summary_length']
    text_features = ['clean_text', 'clean_summary']
    
    # Create pipeline
    # didnt rlly have time to twist the paremeters
    print("\nBuilding pipeline")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('text', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=3,
                stop_words='english',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            ), 'clean_text'),
            ('summary', TfidfVectorizer(
                max_features=1500,
                ngram_range=(1, 2),
                min_df=3,
                stop_words='english',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            ), 'clean_summary')
        ],
        n_jobs=-1 # for parrell processing
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0, random_state=RANDOM_SEED))
    ])
    
    # Train model
    print("\nTraining model")
    X = train_features[numerical_features + text_features]
    y = train_features['Score']
    
    pipeline.fit(X, y)
    
    del X, y, train_features
    gc.collect()
    
    # making the predictions
    print("\nMaking predictions")
    all_predictions = []
    
    for start_idx in range(0, len(test_features), 10000):
        end_idx = min(start_idx + 10000, len(test_features))
        batch = test_features.iloc[start_idx:end_idx]
        batch_predictions = pipeline.predict(batch[numerical_features + text_features])
        all_predictions.extend(batch_predictions)
    
    predictions = np.round(np.clip(all_predictions, 1, 5)).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'Score': predictions
    })
    
    print("\nPrediction distribution:")
    print(pd.Series(predictions).value_counts().sort_index())
    
    submission.to_csv("./data/submission.csv", index=False)
    
    minutes_taken = (time() - start_time) / 60
    print(f"\nTotal execution time: {minutes_taken:.2f} minutes")


if __name__ == "__main__":
    main()