import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_sales_data(num_points=365*3):
    """Generate synthetic sales data with trend, seasonality, and noise"""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2021-01-01', periods=num_points, freq='D')
    
    # Generate trend (gradually increasing)
    trend = np.linspace(100, 200, num_points)
    
    # Generate seasonality (yearly pattern)
    seasonality = 30 * np.sin(2 * np.pi * np.arange(num_points) / 365)
    
    # Generate random noise
    noise = np.random.normal(0, 10, num_points)
    
    # Combine components
    sales = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales
    })
    df.set_index('Date', inplace=True)
    
    return df

def create_sequences(data, window_size):
    """
    Create sequences for time series prediction
    
    Args:
        data: numpy array of time series data
        window_size: number of time steps to look back
    
    Returns:
        X: input sequences (samples, window_size, features)
        y: target values (samples,)
    """
    X, y = [], []
    
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    
    return np.array(X), np.array(y)

def prepare_timeseries_data(df, target_column, window_size, train_split=0.8):
    """
    Prepare time series data for RNN training
    
    Args:
        df: DataFrame with time series data
        target_column: name of the target column
        window_size: number of time steps to look back
        train_split: percentage of data for training
    
    Returns:
        dict with prepared data, scaler, and metadata
    """
    # Extract target values
    data = df[target_column].values.reshape(-1, 1)
    
    # Split into train/test
    train_size = int(len(data) * train_split)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Scale data (fit on training data only)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)
    
    # Reshape for RNN input (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'train_size': train_size,
        'original_data': data
    }