import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

def isolation_forest_scorer(estimator, X):
    # score_samples returns lower values for more anomalous points
    # We negate it so higher score = better (like normal scorers)
    return -estimator.score_samples(X)

def prepare_data_for_model(df):
    """
    Prepare dataframe for Isolation Forest by encoding categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with categorical and numeric columns
    
    Returns:
    --------
    pd.DataFrame, dict
        Encoded dataframe and dictionary of label encoders for each categorical column
    """
    df_encoded = df.copy()
    label_encoders = {}
    
    # Encode categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['category']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    return df_encoded, label_encoders


def fit_isolation_forest(df, contamination=0.05, random_state=42):
    """
    Fit Isolation Forest model for anomaly detection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Encoded dataframe (numeric columns only)
    contamination : float
        Expected proportion of outliers (default 5%)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    IsolationForest, np.array
        Fitted model and anomaly predictions (-1 for anomalies, 1 for normal)
    """
    model = IsolationForest(
        contamination=contamination, # This is the expected proportion of outliers in the dataset.
        random_state=random_state,
        n_estimators=100,
        max_samples='auto'
    )
    
    predictions = model.fit_predict(df)
    anomaly_scores = model.score_samples(df)
    
    return model, predictions, anomaly_scores


def add_anomaly_labels(df_original, predictions, anomaly_scores):
    """
    Add anomaly predictions and scores to original dataframe.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataframe
    predictions : np.array
        Predictions from Isolation Forest (-1 for anomaly, 1 for normal)
    anomaly_scores : np.array
        Anomaly scores from model
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added columns 'is_anomaly' and 'anomaly_score'
    """
    df_results = df_original.copy()
    df_results['is_anomaly'] = predictions == -1
    df_results['anomaly_score'] = anomaly_scores
    
    return df_results


def get_anomaly_statistics(df_results):
    """
    Get statistics about detected anomalies.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Dataframe with anomaly predictions
    
    Returns:
    --------
    dict
        Dictionary with anomaly statistics
    """
    total = len(df_results)
    num_anomalies = df_results['is_anomaly'].sum()
    pct_anomalies = (num_anomalies / total) * 100
    
    stats = {
        'total_records': total,
        'num_anomalies': num_anomalies,
        'pct_anomalies': round(pct_anomalies, 2),
        'num_normal': total - num_anomalies,
        'mean_anomaly_score_anomalies': round(df_results[df_results['is_anomaly']]['anomaly_score'].mean(), 4),
        'mean_anomaly_score_normal': round(df_results[~df_results['is_anomaly']]['anomaly_score'].mean(), 4)
    }
    
    return stats


def print_anomaly_statistics(stats):
    """Print formatted anomaly detection statistics."""
    print("=" * 70)
    print("ANOMALY DETECTION RESULTS - ISOLATION FOREST")
    print("=" * 70)
    print(f"\nTotal Records: {stats['total_records']:,}")
    print(f"Anomalies Detected: {stats['num_anomalies']:,} ({stats['pct_anomalies']:.2f}%)")
    print(f"Normal Records: {stats['num_normal']:,} ({100 - stats['pct_anomalies']:.2f}%)")
    print(f"\nMean Anomaly Score (Anomalies): {stats['mean_anomaly_score_anomalies']}")
    print(f"Mean Anomaly Score (Normal): {stats['mean_anomaly_score_normal']}")
    print("=" * 70)


def visualize_anomaly_distribution(df_results):
    """
    Visualize anomaly detection results.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Dataframe with anomaly predictions
    """
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    anomaly_counts = df_results['is_anomaly'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for normal, red for anomaly
    axes[0].bar(['Normal', 'Anomaly'], [anomaly_counts[False], anomaly_counts[True]], color=colors)
    axes[0].set_title('Anomaly vs Normal Records', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Anomaly score distribution
    axes[1].hist(df_results[~df_results['is_anomaly']]['anomaly_score'], bins=50, 
                 label='Normal', alpha=0.7, color='#2ecc71')
    axes[1].hist(df_results[df_results['is_anomaly']]['anomaly_score'], bins=50, 
                 label='Anomaly', alpha=0.7, color='#e74c3c')
    axes[1].set_title('Anomaly Score Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Frequency')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def show_sample_anomalies(df_results, n_samples=5, numeric_cols=None):
    """
    Display sample anomalies and normal records.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Dataframe with anomaly predictions
    n_samples : int
        Number of samples to display
    numeric_cols : list
        List of numeric columns to display
    """
    print("\n" + "=" * 70)
    print(f"SAMPLE ANOMALIES (Top {n_samples} by anomaly score)")
    print("=" * 70)
    
    anomalies = df_results[df_results['is_anomaly']].nlargest(n_samples, 'anomaly_score')
    
    # Select columns to display
    if numeric_cols is None:
        display_cols = [col for col in anomalies.columns if col not in ['is_anomaly', 'anomaly_score']][:8]
    else:
        display_cols = numeric_cols
    
    display_cols += ['anomaly_score']
    print(anomalies[display_cols].to_string())
    
    print("\n" + "=" * 70)
    print(f"SAMPLE NORMAL RECORDS (Random {n_samples} samples)")
    print("=" * 70)
    
    normal = df_results[~df_results['is_anomaly']].sample(n=min(n_samples, len(df_results[~df_results['is_anomaly']])))
    
    print(normal[display_cols].to_string())
    print("=" * 70)

def get_feature_importance_for_anomalies(df_encoded, model):
    """
    Calculate feature importance for anomaly detection using permutation importance.
    
    Parameters:
    -----------
    df_encoded : pd.DataFrame
        Encoded features dataframe
    model : IsolationForest
        Fitted Isolation Forest model
    
    Returns:
    --------
    pd.DataFrame
        Feature importance scores sorted by importance
    """
    # average anomaly score of the 10% most anomalous samples in the dataset
    def anomaly_scorer(estimator, X, y=None):
        scores = estimator.decision_function(X)
        # Return mean of the 10% most anomalous scores
        threshold = np.percentile(scores, 10)
        return scores[scores <= threshold].mean()
    
    # Use permutation_importance with our custom scorer
    result = permutation_importance(
        estimator=model,
        X=df_encoded,
        y=None,
        scoring=anomaly_scorer,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        max_samples=1.0
    )
    
    # Create DataFrame with results
    feature_importance = pd.DataFrame({
        'feature': df_encoded.columns,
        'importance_mean': result.importances_mean,
    })
    
    # Take absolute values for normalization
    # (We care about magnitude of change, not direction)
    feature_importance['importance_abs'] = feature_importance['importance_mean'].abs()
    
    # Normalize to percentage based on absolute values
    total_abs_importance = feature_importance['importance_abs'].sum()
    if total_abs_importance > 0:
        feature_importance['importance_pct'] = (
            feature_importance['importance_abs'] / total_abs_importance * 100
        )
    else:
        feature_importance['importance_pct'] = 0
    
    # Sort by absolute importance
    feature_importance = feature_importance.sort_values(
        'importance_abs', ascending=False
    ).reset_index(drop=True)
    
    return feature_importance


def analyze_anomaly_characteristics(df_results, df_features):
    """
    Compare mean characteristics of ALL anomalies vs ALL normal records.
    """
    # Use ALL anomalies, not just top N
    anomaly_idx = df_results[df_results['is_anomaly']].index
    normal_idx = df_results[~df_results['is_anomaly']].index
    
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    categorical_cols = df_features.select_dtypes(include=['category', 'object']).columns
    
    comparisons = []
    
    # Numeric features
    for col in numeric_cols:
        anomaly_mean = df_features.loc[anomaly_idx, col].mean()
        normal_mean = df_features.loc[normal_idx, col].mean()
        
        comparisons.append({
            'feature': col,
            'type': 'numeric',
            'anomaly_mean': anomaly_mean,
            'normal_mean': normal_mean,
            'abs_difference': abs(anomaly_mean - normal_mean),
        })
    
    # Categorical features - compare distributions
    for col in categorical_cols:
        anomaly_dist = df_features.loc[anomaly_idx, col].value_counts(normalize=True)
        normal_dist = df_features.loc[normal_idx, col].value_counts(normalize=True)
        
        # Calculate total variation distance or KL divergence
        # For simplicity, just show modes
        anomaly_mode = anomaly_dist.index[0] if len(anomaly_dist) > 0 else None
        normal_mode = normal_dist.index[0] if len(normal_dist) > 0 else None

        # Calculate distribution difference (e.g., total variation distance)
        all_categories = set(anomaly_dist.index) | set(normal_dist.index)
        tv_distance = sum(abs(anomaly_dist.get(cat, 0) - normal_dist.get(cat, 0)) 
                        for cat in all_categories) / 2
        
        comparisons.append({
            'feature': col,
            'type': 'categorical',
            'anomaly_mode': str(anomaly_mode),
            'normal_mode': str(normal_mode),
            'modes_differ': anomaly_mode != normal_mode,
            'distribution_distance': round(tv_distance, 3)
        })

    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df["distance"] = np.where(
        comparisons_df['type'] == 'numeric',
        comparisons_df['abs_difference'],
        comparisons_df['distribution_distance']
    )
    comparisons_df.sort_values('distance', ascending=False, inplace=True)
    comparisons_df.reset_index(drop=True, inplace=True)
    
    return comparisons_df



def visualize_feature_importance(feature_importance, df_features, top_n=15):
    """
    Visualize feature importance for anomaly detection.
    
    Parameters:
    -----------
    feature_importance : pd.DataFrame
        Feature importance dataframe with columns: feature, importance_mean
    df_features : pd.DataFrame
        Original features dataframe (to determine feature types)
    top_n : int
        Number of top features to display
    """
    top_features = feature_importance.head(top_n).copy()
    
    # Determine feature types
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    categorical_cols = df_features.select_dtypes(include=['category', 'object']).columns
    
    # Add feature type to the labels
    feature_labels = []
    for feature in top_features['feature']:
        if feature in numeric_cols:
            feature_labels.append(f"{feature} (numeric)")
        elif feature in categorical_cols:
            feature_labels.append(f"{feature} (categorical)")
        else:
            feature_labels.append(f"{feature} (unknown)")
    
    plt.figure(figsize=(10, max(6, len(top_features) * 0.4)))
    plt.barh(range(len(top_features)), top_features['importance_pct'], color='steelblue')
    plt.yticks(range(len(top_features)), feature_labels)
    plt.xlabel('Importance Score (%):', fontsize=11)
    plt.title(f'Top {top_n} Features for Anomaly Detection', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_anomaly_characteristics(comparison, top_n=15):
    """
    Visualize how anomalies differ from normal records by feature.
    Now handles both numeric and categorical features.
    
    Parameters:
    -----------
    comparison : pd.DataFrame
        Comparison dataframe from analyze_anomaly_characteristics
    top_n : int
        Number of top differing features to display
    """
    top_comparison = comparison.head(top_n)
    
    # Separate numeric and categorical
    numeric_comparison = top_comparison[top_comparison['type'] == 'numeric']
    categorical_comparison = top_comparison[top_comparison['type'] == 'categorical']
    
    # Plot numeric features if any exist
    if len(numeric_comparison) > 0:
        _, ax = plt.subplots(figsize=(12, max(5, len(numeric_comparison) * 0.2)))
        
        x = np.arange(len(numeric_comparison))
        width = 0.35
        
        ax.barh(x - width/2, numeric_comparison['anomaly_mean'], width, 
                label='Anomaly Mean', color='#e74c3c')
        ax.barh(x + width/2, numeric_comparison['normal_mean'], width, 
                label='Normal Mean', color='#2ecc71')
        ax.set_yticks(x)
        ax.set_yticklabels(numeric_comparison['feature'])
        ax.set_xlabel('Anomaly/Normal Mean Value', fontsize=11)
        ax.set_title(f'Numeric Features: Anomaly vs Normal Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
    print(numeric_comparison)
    
    # Print categorical features info
    if len(categorical_comparison) > 0:
        print("\n" + "=" * 80)
        print("CATEGORICAL FEATURES - MOST COMMON VALUES")
        print("=" * 80)
        for _, row in categorical_comparison.iterrows():
            print(f"\n{row['feature']}:")
            print(f"  Anomaly Mode: {row['anomaly_mode']}")
            print(f"  Normal Mode:  {row['normal_mode']}")
            print(f"  TV Difference:   {row['distribution_distance']:+.3f} concentration in anomalies")
        print("=" * 80)


def show_detailed_anomaly_analysis(df_results, df_features, comparison, n_anomalies=5):
    """
    Show detailed analysis of specific anomalies with their distinguishing features.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Results with anomaly labels
    df_features : pd.DataFrame
        Original features
    comparison : pd.DataFrame
        Feature comparison dataframe
    n_anomalies : int
        Number of anomalies to analyze
    """
    print("\n" + "=" * 100)
    print("DETAILED ANOMALY ANALYSIS - DISTINGUISHING FEATURES")
    print("=" * 100)
    
    # Get top anomalies
    top_anomalies = df_results[df_results['is_anomaly']].nsmallest(n_anomalies, 'anomaly_score')
    
    # Get most impactful features (largest % difference)
    top_differing_features = comparison.head(10)['feature'].tolist()
    
    # Get indices of all normal records (once, outside the loop)
    normal_idx = df_results[~df_results['is_anomaly']].index
    
    for idx, (i, anomaly) in enumerate(top_anomalies.iterrows(), 1):
        print(f"\n{'─' * 100}")
        print(f"ANOMALY #{idx} (Anomaly Score: {anomaly['anomaly_score']:.4f})")
        print(f"{'─' * 100}")
        
        print("\nMost Distinguishing Features:")
        for feature in top_differing_features:
            if feature in df_features.columns:
                anomaly_val = df_features.loc[i, feature]
                
                # Get the feature type from comparison
                feature_info = comparison[comparison['feature'] == feature]
                if len(feature_info) == 0:
                    continue
                    
                feature_type = feature_info['type'].values[0]
                diff = feature_info['distance'].values[0]
                
                # Handle numeric vs categorical differently
                if feature_type == 'numeric':
                    normal_mean = df_features.loc[normal_idx, feature].mean()
                    print(f"  {feature:30} | Anomaly: {anomaly_val:>10.2f} | Normal Mean: {normal_mean:>10.2f} | Diff: {diff:+7.3f}")
                else:  # categorical
                    normal_mode = feature_info['normal_mode'].values[0]
                    print(f"  {feature:30} | Anomaly: {str(anomaly_val)[:15]:>15} | Normal Mode: {str(normal_mode)[:15]:>15} | Diff: {diff:+7.3f}")
    
    print("\n" + "=" * 100)