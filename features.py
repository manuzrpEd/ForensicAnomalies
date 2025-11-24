import matplotlib.pyplot as plt
import pandas as pd
from utils import COLS_TO_DROP, GROUPING_CONFIG

def generate_features(df):
    """
    Generate features for anomaly detection.
    Remove non-predictive columns, create temporal features, and encode variables.
    """
    df = df.copy()
    
    # Create temporal features from date_occ
    if 'date_occ' in df.columns:
        df['date_occ'] = pd.to_datetime(df['date_occ'], errors='coerce')
        df['month'] = df['date_occ'].dt.month
        df['day_of_week'] = df['date_occ'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df = df.drop(columns=['date_occ','day_of_week'])

    # Bin vict_age into age groups
    if 'vict_age' in df.columns:
        df['vict_age'] = pd.cut(df['vict_age'], 
                                bins=[18, 30, 45, 60, 75, float('inf')],
                                labels=['18-30', '30-45', '45-60', '60-75', '75+'],
                                right=False)
        df['vict_age'] = df['vict_age'].astype('category')

    # Group vict_sex: keep M, F, and group rest as "Other"
    if 'vict_sex' in df.columns:
        df['vict_sex'] = df['vict_sex'].apply(lambda x: x if x in ['M', 'F'] else 'Other')
        df['vict_sex'] = df['vict_sex'].astype('category')
    
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Encode boolean columns as integers (0/1)
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Drop non-predictive columns (identifiers, case numbers, etc.)
    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns])

    # Group residual categories
    df = group_residual_categories(df)

    
    return df

def plot_feature_distributions(df):
    """
    Plot value counts for all columns in the dataframe.
    Handles categorical, numeric, and boolean columns appropriately.
    """
    cols = df.columns
    n_cols = len(cols)
    
    # Calculate grid dimensions
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    _, axes = plt.subplots(n_rows, 3, figsize=(24, 8 * n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(cols):
        ax = axes[idx]
        
        # Handle different data types
        if df[col].dtype == 'category':
            value_counts = df[col].value_counts(dropna=False).head(20)
            ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels([str(x) if x is not None else 'Missing' for x in value_counts.index],
                             rotation=45, ha='right', fontsize=9)
        elif df[col].dtype in ['float64']:
            ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            ax.set_ylabel('Frequency')
        elif df[col].dtype in ['int64', 'float64', 'Int64', "int"]:  # Boolean encoded as int
            value_counts = df[col].value_counts()
            ax.bar(value_counts.index, value_counts.values)
            ax.set_xticks(sorted(value_counts.index))
            ax.set_xticklabels([str(int(x)) for x in sorted(value_counts.index)], fontsize=11)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        else:
            value_counts = df[col].value_counts(dropna=False).head(20)
            ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=9)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.set_title(f'{col}\n(dtype: {df[col].dtype})', fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def group_residual_categories(df):
    """
    Group infrequent or specified values into residual categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with grouped residual categories
    """
    df = df.copy()
    
    for col, config in GROUPING_CONFIG.items():
        if col not in df.columns:
            continue
        
        # For numeric columns: group by threshold
        if 'threshold' in config and df[col].dtype in ['int64', 'float64', 'Int64', 'int']:
            threshold = config['threshold']
            group_name = config.get('group_name', '3+')
            df[col] = df[col].apply(lambda x: group_name if x >= threshold else x)
            # Convert back to category if needed
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        
        # For categorical columns: keep top N and group rest as "Other"
        elif 'top_n' in config and df[col].dtype == 'category':
            top_n = config['top_n']
            group_name = config.get('group_name', 'Other')
            value_counts = df[col].value_counts()
            top_categories = value_counts.head(top_n).index.tolist()
            df[col] = df[col].apply(lambda x: x if x in top_categories else group_name)
            df[col] = df[col].astype('category')

    # Auto-group small categories by percentage threshold
    pct_threshold = 0.01  # 1% threshold
    categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
    for col in categorical_cols:
        if 'Missing' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories(['Missing'])
            df[col] = df[col].fillna('Missing')
        value_counts = df[col].value_counts(dropna=False)
        total = value_counts.sum()
        
        # Find categories below 1% threshold
        small_categories = value_counts[value_counts / total < pct_threshold].index.tolist()
        
        if small_categories:
            def group_func(x):
                if x in small_categories:
                    return 'Other'
                return x
            df[col] = df[col].apply(group_func)
            df[col] = df[col].astype('category')
    # Remove unused categories with 0 counts
    categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].cat.remove_unused_categories()
    
    return df

def check_data_quality(df):
    """
    Check if dataframe is balanced with no NaN values and consistent row counts.
    Prints a detailed quality report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to validate
    
    Returns:
    --------
    dict
        Dictionary with quality checks
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'nan_summary': {},
        'is_balanced': True,
        'issues': []
    }
    
    # Check for NaN values in each column
    for col in df.columns:
        nan_count = df[col].isna().sum()
        nan_pct = (nan_count / len(df)) * 100
        
        report['nan_summary'][col] = {
            'nan_count': nan_count,
            'nan_percentage': round(nan_pct, 2),
            'non_nan_count': len(df) - nan_count
        }
        
        if nan_count > 0:
            report['is_balanced'] = False
            report['issues'].append(f"{col}: {nan_count} NaN values ({nan_pct:.2f}%)")
    
    # Check if all columns have same number of non-NaN entries
    non_nan_counts = {col: (df[col].notna().sum()) for col in df.columns}
    unique_counts = set(non_nan_counts.values())
    
    if len(unique_counts) > 1:
        report['is_balanced'] = False
        report['issues'].append(f"Unequal non-NaN counts: {non_nan_counts}")
    
    # Print report
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"\nTotal rows: {report['total_rows']}")
    print(f"Total columns: {report['total_columns']}")
    print(f"\nIs Balanced: {report['is_balanced']}")

    print(f"\nNaN Summary:")
    print("-" * 60)
    for col, stats in report['nan_summary'].items():
        print(f"{col:30} | NaN: {stats['nan_count']:6} ({stats['nan_percentage']:5.2f}%) | Non-NaN: {stats['non_nan_count']:6}")

    if report['issues']:
        print(f"\nIssues Found:")
        print("-" * 60)
        for issue in report['issues']:
            print(f"  ⚠ {issue}")
    else:
        print(f"\n✓ No issues found - Data is balanced!")
    
    return report