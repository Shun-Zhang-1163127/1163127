#!/usr/bin/env python3
"""
Test script for Feature Engineering functions
COMP647 Assignment 03 - Student ID: 1163127

This script tests all feature engineering functions to ensure they work correctly
with the actual Lending Club dataset.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
import category_encoders as ce
from imblearn.over_sampling import SMOTE
from collections import Counter

def test_data_loading():
    """Test if data can be loaded successfully"""
    print("=" * 50)
    print("TEST 1: Data Loading")
    print("=" * 50)

    try:
        df = pd.read_csv('data/processed/accepted_sample_10000.csv')
        print(f"[PASS] Data loaded successfully: {df.shape}")
        print(f"[PASS] Columns: {len(df.columns)}")

        # Check for categorical and numerical features
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

        print(f"[PASS] Categorical features: {len(categorical_features)}")
        print(f"[PASS] Numerical features: {len(numerical_features)}")

        return df, categorical_features, numerical_features

    except FileNotFoundError:
        print("[FAIL] Data file not found. Please run Assignment 02 notebooks first.")
        return None, [], []
    except Exception as e:
        print(f"[FAIL] Error loading data: {e}")
        return None, [], []

def test_label_encoding(df, categorical_features):
    """Test label encoding functionality"""
    print("\n" + "=" * 50)
    print("TEST 2: Label Encoding")
    print("=" * 50)

    if df is None or not categorical_features:
        print("[FAIL] Skipping - no data or categorical features")
        return False

    try:
        # Test with first categorical feature
        test_feature = categorical_features[0]
        original_unique = df[test_feature].nunique()

        # Apply label encoding
        le = LabelEncoder()
        df_test = df.copy()
        df_test[test_feature] = df_test[test_feature].fillna('Unknown')
        encoded_values = le.fit_transform(df_test[test_feature])

        # Validate results
        assert len(encoded_values) == len(df), "Encoded length mismatch"
        assert max(encoded_values) == original_unique - 1, "Max encoded value incorrect"
        assert min(encoded_values) == 0, "Min encoded value should be 0"

        print(f"[PASS] Label encoding successful for '{test_feature}'")
        print(f"[PASS] {original_unique} categories -> 0 to {max(encoded_values)}")
        return True

    except Exception as e:
        print(f"[FAIL] Label encoding failed: {e}")
        return False

def test_onehot_encoding(df, categorical_features):
    """Test one-hot encoding functionality"""
    print("\n" + "=" * 50)
    print("TEST 3: One-Hot Encoding")
    print("=" * 50)

    if df is None or not categorical_features:
        print("[FAIL] Skipping - no data or categorical features")
        return False

    try:
        # Find a feature with reasonable cardinality
        suitable_feature = None
        for col in categorical_features:
            if df[col].nunique() <= 10:
                suitable_feature = col
                break

        if not suitable_feature:
            print("[FAIL] No suitable feature found (need ≤10 categories)")
            return False

        original_categories = df[suitable_feature].nunique()

        # Apply one-hot encoding
        dummies = pd.get_dummies(df[suitable_feature], prefix=suitable_feature, drop_first=True)

        # Validate results
        expected_columns = original_categories - 1  # drop_first=True
        assert len(dummies.columns) == expected_columns, f"Expected {expected_columns} columns, got {len(dummies.columns)}"
        assert dummies.dtypes.all() in [bool, int, np.uint8], "Dummy variables should be binary"

        print(f"[PASS] One-hot encoding successful for '{suitable_feature}'")
        print(f"[PASS] {original_categories} categories -> {len(dummies.columns)} dummy columns")
        return True

    except Exception as e:
        print(f"[FAIL] One-hot encoding failed: {e}")
        return False

def test_binary_encoding(df, categorical_features):
    """Test binary encoding functionality"""
    print("\n" + "=" * 50)
    print("TEST 4: Binary Encoding")
    print("=" * 50)

    if df is None or not categorical_features:
        print("[FAIL] Skipping - no data or categorical features")
        return False

    try:
        # Find a feature with high cardinality
        suitable_feature = None
        for col in categorical_features:
            if 5 < df[col].nunique() <= 50:
                suitable_feature = col
                break

        if not suitable_feature:
            # Use first categorical feature anyway for testing
            suitable_feature = categorical_features[0]

        original_categories = df[suitable_feature].nunique()

        # Apply binary encoding
        df_test = df.copy()
        df_test[suitable_feature] = df_test[suitable_feature].fillna('Unknown')
        encoder = ce.BinaryEncoder(cols=[suitable_feature])
        encoded_df = encoder.fit_transform(df_test[[suitable_feature]])

        # Validate results
        expected_columns = int(np.ceil(np.log2(original_categories)))
        assert len(encoded_df.columns) <= expected_columns + 1, "Too many binary columns"
        assert all(encoded_df.dtypes == int), "Binary encoded values should be integers"

        print(f"[PASS] Binary encoding successful for '{suitable_feature}'")
        print(f"[PASS] {original_categories} categories -> {len(encoded_df.columns)} binary columns")
        return True

    except Exception as e:
        print(f"[FAIL] Binary encoding failed: {e}")
        return False

def test_minmax_scaling(df, numerical_features):
    """Test min-max scaling functionality"""
    print("\n" + "=" * 50)
    print("TEST 5: Min-Max Scaling")
    print("=" * 50)

    if df is None or not numerical_features:
        print("[FAIL] Skipping - no data or numerical features")
        return False

    try:
        # Select features for testing (exclude ID columns)
        test_features = [col for col in numerical_features[:5]
                        if not col.lower().endswith('_id') and df[col].nunique() > 2]

        if not test_features:
            print("[FAIL] No suitable numerical features found")
            return False

        # Apply min-max scaling
        scaler = MinMaxScaler()
        df_test = df[test_features].fillna(df[test_features].mean())
        scaled_data = scaler.fit_transform(df_test)

        # Validate results
        assert scaled_data.min() >= 0, "Min-max scaled values should be >= 0"
        assert scaled_data.max() <= 1, "Min-max scaled values should be <= 1"
        assert scaled_data.shape == df_test.shape, "Shape should be preserved"

        print(f"[PASS] Min-max scaling successful for {len(test_features)} features")
        print(f"[PASS] Range: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
        return True

    except Exception as e:
        print(f"[FAIL] Min-max scaling failed: {e}")
        return False

def test_standard_scaling(df, numerical_features):
    """Test standard scaling functionality"""
    print("\n" + "=" * 50)
    print("TEST 6: Standard Scaling")
    print("=" * 50)

    if df is None or not numerical_features:
        print("[FAIL] Skipping - no data or numerical features")
        return False

    try:
        # Select features for testing
        test_features = [col for col in numerical_features[:5]
                        if not col.lower().endswith('_id') and df[col].nunique() > 2]

        if not test_features:
            print("[FAIL] No suitable numerical features found")
            return False

        # Apply standard scaling
        scaler = StandardScaler()
        df_test = df[test_features].fillna(df[test_features].mean())
        scaled_data = scaler.fit_transform(df_test)

        # Validate results (allow small numerical errors)
        mean_check = abs(scaled_data.mean()) < 1e-10
        std_check = abs(scaled_data.std() - 1) < 1e-10

        assert mean_check, f"Mean should be ~0, got {scaled_data.mean()}"
        assert std_check, f"Std should be ~1, got {scaled_data.std()}"
        assert scaled_data.shape == df_test.shape, "Shape should be preserved"

        print(f"[PASS] Standard scaling successful for {len(test_features)} features")
        print(f"[PASS] Mean: {scaled_data.mean():.6f}, Std: {scaled_data.std():.6f}")
        return True

    except Exception as e:
        print(f"[FAIL] Standard scaling failed: {e}")
        return False

def test_smote(df, numerical_features):
    """Test SMOTE functionality"""
    print("\n" + "=" * 50)
    print("TEST 7: SMOTE Class Balancing")
    print("=" * 50)

    if df is None or not numerical_features:
        print("[FAIL] Skipping - no data or numerical features")
        return False

    try:
        # Create a binary target for testing
        target_col = 'loan_status' if 'loan_status' in df.columns else None
        if not target_col:
            # Create synthetic binary target
            y = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        else:
            if df[target_col].nunique() == 2:
                y = df[target_col]
            else:
                # Convert to binary
                y = (df[target_col] == df[target_col].mode()[0]).astype(int)

        # Select numerical features
        test_features = [col for col in numerical_features[:5]
                        if not col.lower().endswith('_id') and df[col].nunique() > 2]

        if len(test_features) < 2:
            print("[FAIL] Need at least 2 numerical features for SMOTE")
            return False

        X = df[test_features].fillna(df[test_features].mean())

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Validate results
        original_counts = Counter(y)
        resampled_counts = Counter(y_resampled)

        assert len(X_resampled) >= len(X), "SMOTE should increase or maintain sample size"
        assert len(y_resampled) == len(X_resampled), "X and y should have same length"
        assert len(set(y_resampled)) == len(set(y)), "Classes should be preserved"

        print(f"[PASS] SMOTE successful")
        print(f"[PASS] Original distribution: {dict(original_counts)}")
        print(f"[PASS] Resampled distribution: {dict(resampled_counts)}")
        print(f"[PASS] Sample size: {len(y)} -> {len(y_resampled)}")
        return True

    except Exception as e:
        print(f"[FAIL] SMOTE failed: {e}")
        return False

def main():
    """Run all feature engineering tests"""
    print("FEATURE ENGINEERING TEST SUITE")
    print("COMP647 Assignment 03 - Student ID: 1163127")
    print("Testing all feature engineering functions...")

    # Load data
    df, categorical_features, numerical_features = test_data_loading()

    # Run all tests
    tests = [
        test_label_encoding(df, categorical_features),
        test_onehot_encoding(df, categorical_features),
        test_binary_encoding(df, categorical_features),
        test_minmax_scaling(df, numerical_features),
        test_standard_scaling(df, numerical_features),
        test_smote(df, numerical_features)
    ]

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(tests)
    total = len(tests)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED - Feature engineering functions are working correctly!")
        return True
    else:
        print(f"[WARN] {total - passed} tests failed - Please check the implementations")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)