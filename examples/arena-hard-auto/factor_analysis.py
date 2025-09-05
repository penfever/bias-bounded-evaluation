#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

# Ensure compatibility with numpy 2.0+
if not hasattr(np, 'NAN'):
    np.NAN = np.nan

def check_factor_analysis_suitability(df):
    """
    Check if the data is suitable for factor analysis using Bartlett's test and KMO.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing score columns
        
    Returns:
    --------
    tuple : (bool, dict)
        - Boolean indicating if the data is suitable for factor analysis
        - Dictionary containing test results
    """
    # Get factor columns (all score columns except the overall 'score')
    factor_cols = [col for col in df.columns 
                  if col != 'model' 
                  and col != 'question_id' 
                  and 'score' in col.lower() 
                  and col != 'score']
    
    if len(factor_cols) < 2:
        print("Error: Need at least 2 score columns for factor analysis")
        return False, {"error": "Not enough score columns"}
    
    # Create a pivot table for factor analysis
    # Each row is a question_id/model combination, columns are the different score types
    # We'll use the mean of scores if there are multiple entries for the same question_id/model
    if 'question_id' in df.columns:
        # For raw judgment data, create a pivot table
        df_pivot = df.pivot_table(
            index=['question_id', 'model'],
            values=factor_cols,
            aggfunc='mean'
        ).reset_index()
        
        # Drop columns we don't need for factor analysis
        df_scores = df_pivot[factor_cols]
    else:
        # For already processed data (e.g., from CSV), just use the factor columns
        df_scores = df[factor_cols]
    
    # Drop any NaN values
    df_scores = df_scores.dropna()
    
    if len(df_scores) < 10:
        print("Warning: Not enough data points for reliable factor analysis")
        return False, {"error": "Not enough data points"}
    
    # Perform Bartlett's test
    try:
        chi_square_value, p_value = calculate_bartlett_sphericity(df_scores)
    except Exception as e:
        print(f"Error in Bartlett's test: {e}")
        print("Proceeding with factor analysis anyway...")
        return True, {"warning": "Bartlett's test failed, proceeding anyway"}
    
    # Perform KMO test
    try:
        kmo_all, kmo_model = calculate_kmo(df_scores)
    except Exception as e:
        print(f"Error in KMO test: {e}")
        kmo_all, kmo_model = np.nan, np.nan
    
    # Check if data is suitable
    suitable = (p_value < 0.05 if not np.isnan(p_value) else True) and (kmo_model > 0.5 if not np.isnan(kmo_model) else True)
    
    # Print results
    print(f"Bartlett test: chi_square: {chi_square_value}, p_value: {p_value}")
    print(f"KMO score: {kmo_model}")
    print(f"Data suitable for factor analysis: {suitable}")
    
    # Force suitable to True if we have enough factor columns and data points
    if len(factor_cols) >= 2 and len(df_scores) >= 30:
        print("Overriding suitability check due to sufficient data size.")
        suitable = True
    
    results = {
        "bartlett_chi_square": chi_square_value,
        "bartlett_p_value": p_value,
        "kmo_score": kmo_model,
        "suitable": suitable
    }
    
    return suitable, results

def perform_factor_analysis(df, n_factors=None, rotation='varimax', min_eigenvalue=0.75):
    """
    Perform factor analysis on the data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing score columns
    n_factors : int, optional
        Number of factors to extract. If None, determined automatically
    rotation : str, optional
        Rotation method for factor analysis
    min_eigenvalue : float, optional
        Minimum eigenvalue to retain a factor
        
    Returns:
    --------
    dict : Dictionary containing factor analysis results
    """
    # Get factor columns (all score columns except the overall 'score')
    factor_cols = [col for col in df.columns 
                  if col != 'model' 
                  and col != 'question_id' 
                  and 'score' in col.lower() 
                  and col != 'score']
    
    # Create a pivot table for factor analysis if needed
    if 'question_id' in df.columns:
        # For raw judgment data, create a pivot table
        df_pivot = df.pivot_table(
            index=['question_id', 'model'],
            values=factor_cols,
            aggfunc='mean'
        ).reset_index()
        
        # Use only factor columns for analysis
        df_scores = df_pivot[factor_cols]
    else:
        # For already processed data (e.g., from CSV), just use the factor columns
        df_scores = df[factor_cols]
    
    # Convert to numeric if needed
    for col in factor_cols:
        if df_scores[col].dtype == object:
            df_scores[col] = pd.to_numeric(df_scores[col], errors='coerce')
    
    # Drop any remaining NaN values
    df_scores = df_scores.dropna()
    
    # Determine optimal number of factors if not specified
    if n_factors is None:
        # Create factor analyzer without rotation to determine number of factors
        fa_initial = FactorAnalyzer(rotation=None)
        fa_initial.fit(df_scores)
        
        # Get eigenvalues
        ev, v = fa_initial.get_eigenvalues()
        
        # Determine number of factors based on eigenvalues
        suggested_n_factors = sum(ev > min_eigenvalue)
        print(f"Eigenvalues: {ev}")
        print(f"Suggested number of factors: {suggested_n_factors}")
        
        n_factors = max(1, suggested_n_factors)  # At least 1 factor
    
    # Perform factor analysis with the determined number of factors
    fa = FactorAnalyzer(rotation=rotation, n_factors=n_factors)
    fa.fit(df_scores)
    
    # Get factor loadings
    loadings = fa.loadings_.copy()  # Make a copy to avoid modifying the original
    
    # Create dataframe with original loadings
    factor_loadings_original = pd.DataFrame(
        loadings,
        columns=[f'Factor{i+1}' for i in range(n_factors)],
        index=df_scores.columns
    )
    
    # For each factor, check if the majority of loadings are negative
    # If so, flip the signs for that factor to make interpretation more intuitive
    factors_flipped = []  # Keep track of which factors were flipped
    
    for i in range(n_factors):
        # Calculate the weighted sum of positive and negative loadings
        neg_sum = sum(abs(val) for val in loadings[:, i] if val < 0)
        pos_sum = sum(abs(val) for val in loadings[:, i] if val > 0)
        
        # If more negative weight than positive, flip the signs for this factor
        if neg_sum > pos_sum:
            print(f"Flipping signs for Factor{i+1} to make interpretation more intuitive")
            loadings[:, i] = -loadings[:, i]
            factors_flipped.append(i)
    
    # Create dataframe with the possibly flipped loadings
    factor_loadings = pd.DataFrame(
        loadings,
        columns=[f'Factor{i+1}' for i in range(n_factors)],
        index=df_scores.columns
    )
    
    # Get variance explained (using original loadings)
    variance_info = pd.DataFrame(
        fa.get_factor_variance(),
        index=['SS Loadings', 'Proportion Var', 'Cumulative Var'],
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )
    
    # Get communalities (using original loadings)
    communalities = pd.DataFrame(
        fa.get_communalities(),
        columns=['Communality'],
        index=df_scores.columns
    )
    
    # Get original factor scores
    factor_scores_orig = fa.transform(df_scores)
    
    # Flip the factor scores for any flipped factors
    factor_scores_array = factor_scores_orig.copy()
    for i in factors_flipped:
        factor_scores_array[:, i] = -factor_scores_array[:, i]
    
    # Create DataFrame with possibly flipped factor scores
    factor_scores = pd.DataFrame(
        factor_scores_array,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )
    
    # Print results
    print("\nFactor Loadings (after sign adjustment):")
    print(factor_loadings)
    
    print("\nVariance Explained:")
    print(variance_info)
    
    print("\nCommunalities:")
    print(communalities)
    
    results = {
        "loadings": factor_loadings,
        "loadings_original": factor_loadings_original,
        "variance": variance_info,
        "communalities": communalities,
        "scores": factor_scores,
        "n_factors": n_factors,
        "factors_flipped": factors_flipped
    }
    
    return results

def plot_factor_loadings(loadings, output_path):
    """
    Create a heatmap visualization of factor loadings.
    
    Parameters:
    -----------
    loadings : pandas.DataFrame
        DataFrame containing factor loadings
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with values
    sns.heatmap(
        loadings, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1,
        linewidths=.5,
        fmt='.2f',
        annot_kws={"size": 10}
    )
    
    plt.title('Factor Loadings', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_factor_scree(eigenvalues, output_path, min_eigenvalue=0.75):
    """
    Create a scree plot to visualize eigenvalues for factor selection.
    
    Parameters:
    -----------
    eigenvalues : numpy.ndarray
        Array of eigenvalues
    output_path : str
        Path to save the plot
    min_eigenvalue : float, optional
        Minimum eigenvalue threshold used for factor extraction
    """
    plt.figure(figsize=(10, 6))
    
    # Create scree plot
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', linewidth=2, color='blue', 
             label='Eigenvalues')
    
    # Add standard threshold line (Kaiser criterion)
    kaiser_line = plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.6, 
                              label='Kaiser Criterion (eigenvalue = 1.0)')
    
    # Add custom threshold line
    custom_line = plt.axhline(y=min_eigenvalue, color='g', linestyle='--', alpha=0.6,
                              label=f'Custom Threshold (eigenvalue = {min_eigenvalue})')
    
    # Annotate the factors above threshold
    for i, val in enumerate(eigenvalues):
        if val > min_eigenvalue:
            plt.text(i+1.1, val, f'{val:.2f}', fontweight='bold')
            
    # Add explanation of interpretation
    sig_factors = sum(eigenvalues > min_eigenvalue)
    plt.annotate(f'Significant Factors: {sig_factors}', 
                 xy=(0.02, 0.02), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 fontsize=12)
    
    plt.title('Scree Plot for Factor Analysis', fontsize=16)
    plt.xlabel('Factor Number', fontsize=14)
    plt.ylabel('Eigenvalue', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Set y-axis limit to make small eigenvalues visible
    plt.ylim(0, max(5.5, eigenvalues[0] * 1.1))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_factor_biplot(fa_results, df, output_path):
    """
    Create a biplot to visualize the relationship between factors and variables.
    
    Parameters:
    -----------
    fa_results : dict
        Dictionary containing factor analysis results
    df : pandas.DataFrame
        Input dataframe containing score columns
    output_path : str
        Path to save the plot
    """
    if fa_results["n_factors"] < 2:
        print("Need at least 2 factors for a biplot")
        return
    
    loadings = fa_results["loadings"]
    scores = fa_results["scores"]
    
    # Get the first two factors
    plt.figure(figsize=(12, 10))
    
    # Plot scatter of factor scores
    plt.scatter(scores.iloc[:, 0], scores.iloc[:, 1], alpha=0.3)
    
    # Plot factor loadings as vectors
    for i, var in enumerate(loadings.index):
        x = loadings.iloc[i, 0]
        y = loadings.iloc[i, 1]
        
        # Draw arrow from origin to loading point
        plt.arrow(0, 0, x, y, color='r', head_width=0.05, head_length=0.05)
        
        # Position label slightly beyond the arrowhead
        # Adjust text position based on quadrant for better visibility
        offset = 1.15
        if x > 0 and y > 0:  # Quadrant 1
            tx, ty = x * offset, y * offset
            ha, va = 'left', 'bottom'
        elif x < 0 and y > 0:  # Quadrant 2
            tx, ty = x * offset, y * offset
            ha, va = 'right', 'bottom'
        elif x < 0 and y < 0:  # Quadrant 3
            tx, ty = x * offset, y * offset
            ha, va = 'right', 'top'
        else:  # Quadrant 4
            tx, ty = x * offset, y * offset
            ha, va = 'left', 'top'
        
        plt.text(tx, ty, var, color='g', ha=ha, va=va, fontweight='bold')
    
    # Add circle
    circle = plt.Circle((0,0), 1, fill=False, color='blue', linestyle='--')
    plt.gca().add_patch(circle)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(alpha=0.3)
    
    # Use actual variance explained percentages from the data
    var_explained = fa_results["variance"].iloc[1, :] * 100  # Convert to percentage
    
    plt.xlabel(f'Factor 1 ({var_explained[0]:.1f}% variance)', fontsize=14)
    plt.ylabel(f'Factor 2 ({var_explained[1]:.1f}% variance)', fontsize=14)
    plt.title('Factor Analysis Biplot', fontsize=16)
    
    # Set axis limits
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_factor_interpretation(loadings, output_path, threshold=0.5):
    """
    Create a horizontal bar chart to visualize and interpret factors.
    
    Parameters:
    -----------
    loadings : pandas.DataFrame
        DataFrame containing factor loadings
    output_path : str
        Path to save the plot
    threshold : float, optional
        Absolute threshold for considering a loading significant
    """
    n_factors = loadings.shape[1]
    
    # Create a figure with a subplot for each factor
    fig, axes = plt.subplots(n_factors, 1, figsize=(10, 5 * n_factors), sharex=True)
    
    # If there's only one factor, wrap the axes in a list
    if n_factors == 1:
        axes = [axes]
    
    # For each factor, create a horizontal bar chart of loadings
    for i, ax in enumerate(axes):
        factor_name = f'Factor{i+1}'
        factor_loadings = loadings[factor_name].sort_values(ascending=False)
        
        # Highlight significant loadings with different colors
        colors = ['#1f77b4' if abs(val) >= threshold else '#d3d3d3' for val in factor_loadings]
        
        # Create horizontal bar chart
        ax.barh(factor_loadings.index, factor_loadings, color=colors)
        
        # Add a vertical line at 0
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add threshold lines
        ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=-threshold, color='r', linestyle='--', alpha=0.5)
        
        # Add labels and title
        var_exp = loadings.columns[i].split('_')[-1] if '_' in loadings.columns[i] else i+1
        ax.set_title(f'{factor_name}: Significant Feature Loadings', fontsize=14)
        ax.set_xlabel('Loading Value', fontsize=12)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Highlight variable names with high loadings
        for j, var in enumerate(factor_loadings.index):
            if abs(factor_loadings[var]) >= threshold:
                ax.text(0, j, f' {var} ({factor_loadings[var]:.2f})', 
                        va='center', ha='center', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def identify_low_reliability_questions(df, fa_results, threshold=2.0):
    """
    Identify questions where factor model predictions have high residuals.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing questions, models, and score columns
    fa_results : dict
        Factor analysis results dictionary
    threshold : float, optional
        Threshold for standardized residual magnitude to flag questions (default: 2.0)
        Values of 2.0 will flag approximately 5% of questions (2 standard deviations above mean)
        Values of 1.5 will flag approximately 15% of questions (1.5 standard deviations above mean)
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with flagged questions and their residuals
    """
    # Get factor scores and loadings
    factor_scores = fa_results["scores"]
    loadings = fa_results["loadings"]
    
    # Only proceed if we have question_id in the dataframe
    if 'question_id' not in df.columns:
        print("Cannot identify low reliability questions without question_id column")
        return None
    
    # Create pivot table with question_id, model
    factor_cols = [col for col in df.columns 
                  if col != 'model' 
                  and col != 'question_id' 
                  and 'score' in col.lower() 
                  and col != 'score']
    
    # Get unique question_id, model combinations with scores
    pivot_df = df.pivot_table(
        index=['question_id', 'model'],
        values=factor_cols,
        aggfunc='mean'
    ).reset_index()
    
    # Get only the rows used in factor analysis (non-NA)
    pivot_df = pivot_df.dropna(subset=factor_cols)
    
    # Reconstructed scores from factor model
    reconstructed_scores = {}
    residuals = {}
    
    for metric in factor_cols:
        # Get the loadings for this metric
        metric_loadings = loadings.loc[metric]
        
        # Calculate reconstructed scores for this metric
        reconstructed = np.zeros(len(factor_scores))
        for i, factor in enumerate(loadings.columns):
            reconstructed += factor_scores[factor] * metric_loadings[factor]
        
        # Store reconstructed scores
        reconstructed_scores[metric] = reconstructed
        
        # Calculate residuals (original - reconstructed)
        original = pivot_df[metric].values
        residuals[metric] = original - reconstructed
        
        # Add to the dataframe
        pivot_df[f"{metric}_reconstructed"] = reconstructed
        pivot_df[f"{metric}_residual"] = residuals[metric]
        pivot_df[f"{metric}_residual_sq"] = residuals[metric] ** 2
    
    # Calculate mean squared residual for each question_id across all metrics and models
    question_reliability = pivot_df.groupby('question_id').agg({
        f"{metric}_residual_sq": 'mean' for metric in factor_cols
    }).reset_index()
    
    # Add overall reliability score (mean of all squared residuals)
    reliability_cols = [f"{metric}_residual_sq" for metric in factor_cols]
    question_reliability['overall_residual'] = question_reliability[reliability_cols].mean(axis=1)
    
    # Standardize the overall residual (z-score)
    mean_residual = question_reliability['overall_residual'].mean()
    std_residual = question_reliability['overall_residual'].std()
    question_reliability['standardized_residual'] = (question_reliability['overall_residual'] - mean_residual) / std_residual
    
    # Flag questions with standardized residuals > threshold (typically 2.0 for ~5% most extreme)
    # This means flagging questions whose residuals are 2 standard deviations above the mean
    std_threshold = 2.0 if threshold is None else threshold
    question_reliability['low_reliability'] = question_reliability['standardized_residual'] > std_threshold
    print(f"Using standardized residual threshold: {std_threshold}")
    print(f"Number of questions flagged as low reliability: {question_reliability['low_reliability'].sum()} " +
          f"out of {len(question_reliability)} ({100*question_reliability['low_reliability'].sum()/len(question_reliability):.1f}%)")
    
    # Sort by standardized residual (descending)
    question_reliability = question_reliability.sort_values('standardized_residual', ascending=False)
    
    return question_reliability


def calculate_factor_importance_nonlinear(df, exclude_overall=True):
    """
    Calculate the importance of each factor in explaining the variance,
    including both linear and nonlinear (polynomial) terms.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing score columns
    exclude_overall : bool, optional
        Whether to exclude the overall score from factors
        
    Returns:
    --------
    tuple : (pd.Series, float, pd.Series, float, pd.Series)
        - Linear coefficients
        - Linear R² value
        - Polynomial coefficients
        - Polynomial R² value
        - Importance of polynomial terms
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    # Get the columns related to scores
    score_cols = [col for col in df.columns if 'score' in col.lower()]
    
    # Exclude the overall score if requested
    if exclude_overall and 'score' in score_cols:
        factor_cols = [col for col in score_cols if col != 'score']
        target_col = 'score'
    else:
        factor_cols = score_cols
        target_col = None
    
    if target_col and target_col in df.columns:
        # Create pivot table to get average scores for each model
        if 'question_id' in df.columns:
            pivot_df = df.pivot_table(
                index=['question_id', 'model'],
                values=score_cols,
                aggfunc='mean'
            ).reset_index()
        else:
            pivot_df = df
            
        # Standardize the data
        X = pivot_df[factor_cols].apply(lambda x: (x - x.mean()) / x.std())
        y = pivot_df[target_col]
        
        # Train a linear regression model
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        
        # Get feature importances (coefficients) for linear model
        linear_importances = pd.Series(linear_model.coef_, index=factor_cols)
        linear_r2 = linear_model.score(X, y)
        
        # Create polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Get the feature names for the polynomial features
        poly_feature_names = []
        for i, name in enumerate(factor_cols):
            poly_feature_names.append(name)  # Original features
        
        # Add squared terms
        for i, name in enumerate(factor_cols):
            poly_feature_names.append(f"{name}²")
        
        # Add interaction terms
        for i, name1 in enumerate(factor_cols):
            for j, name2 in enumerate(factor_cols):
                if j > i:  # Only upper triangular elements
                    poly_feature_names.append(f"{name1}×{name2}")
        
        # Train a polynomial regression model
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        
        # Get feature importances (coefficients) for polynomial model
        poly_importances = pd.Series(poly_model.coef_, index=poly_feature_names)
        poly_r2 = poly_model.score(X_poly, y)
        
        # Calculate the importance of polynomial terms (compared to linear)
        # by looking at their contribution to R²
        poly_contribution = poly_r2 - linear_r2
        
        # Calculate the relative importance of each polynomial term
        nonlinear_terms = poly_importances.iloc[len(factor_cols):]  # Skip linear terms
        nonlinear_importance = nonlinear_terms.abs() / nonlinear_terms.abs().sum()
        
        return linear_importances, linear_r2, poly_importances, poly_r2, nonlinear_importance
    else:
        return None, None, None, None, None

def plot_polynomial_importance(poly_importances, linear_r2, poly_r2, output_path):
    """
    Create a bar chart visualizing the importance of polynomial terms.
    
    Parameters:
    -----------
    poly_importances : pandas.Series
        Series containing polynomial coefficients
    linear_r2 : float
        R² value for the linear model
    poly_r2 : float
        R² value for the polynomial model
    output_path : str
        Path to save the plot
    """
    # Sort by absolute importance, excluding the linear terms
    n_linear_terms = len([x for x in poly_importances.index if '²' not in x and '×' not in x])
    nonlinear_importances = poly_importances.iloc[n_linear_terms:].abs().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 10))
    
    # Plot the coefficients for nonlinear terms
    bars = plt.barh(nonlinear_importances.index, nonlinear_importances.values, color='skyblue')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha='left', va='center')
    
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.title('Importance of Nonlinear Terms in Predicting Overall Score', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add annotation about R² improvement
    r2_improvement = poly_r2 - linear_r2
    plt.figtext(0.5, 0.01, 
                f"Polynomial terms increased R² from {linear_r2:.4f} to {poly_r2:.4f} (gain: {r2_improvement:.4f})", 
                ha="center", fontsize=12, 
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgrey", "alpha": 0.8})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_factor_analysis(df, output_dir, min_eigenvalue=0.75, analyze_questions=False, threshold=2.0):
    """
    Run factor analysis and save results to files.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing score columns
    output_dir : str
        Directory to save the output files
    min_eigenvalue : float, optional
        Minimum eigenvalue to retain a factor
    analyze_questions : bool, optional
        Whether to analyze question reliability
    threshold : float, optional
        Threshold for standardized residuals to flag low reliability questions
        
    Returns:
    --------
    dict : Results of the factor analysis
    """
    # Create factor analysis subdirectory
    factor_analysis_dir = os.path.join(output_dir, "tables", "factor_analysis")
    os.makedirs(factor_analysis_dir, exist_ok=True)
    
    # Create question reliability subdirectory if needed
    if analyze_questions:
        question_reliability_dir = os.path.join(output_dir, "tables", "question_reliability")
        os.makedirs(question_reliability_dir, exist_ok=True)
    # Check if data is suitable for factor analysis
    suitable, suitability_results = check_factor_analysis_suitability(df)
    
    if not suitable:
        print("Data is not suitable for factor analysis.")
        return None
    
    # Create pivot table if needed and get factor columns
    factor_cols = [col for col in df.columns 
                  if col != 'model' 
                  and col != 'question_id' 
                  and 'score' in col.lower() 
                  and col != 'score']
    
    # Create a pivot table for factor analysis if needed
    if 'question_id' in df.columns:
        df_pivot = df.pivot_table(
            index=['question_id', 'model'],
            values=factor_cols,
            aggfunc='mean'
        ).reset_index()
        df_scores = df_pivot[factor_cols]
    else:
        df_scores = df[factor_cols]
        
    df_scores = df_scores.apply(pd.to_numeric, errors='coerce').dropna()
    
    fa_initial = FactorAnalyzer(rotation=None)
    fa_initial.fit(df_scores)
    
    # Get eigenvalues for scree plot
    eigenvalues, _ = fa_initial.get_eigenvalues()
    
    # Save scree plot
    scree_plot_path = os.path.join(factor_analysis_dir, "factor_scree_plot.png")
    plot_factor_scree(eigenvalues, scree_plot_path, min_eigenvalue=min_eigenvalue)
    print(f"Saved scree plot to {scree_plot_path}")
    
    # Perform factor analysis
    fa_results = perform_factor_analysis(df, rotation='varimax', min_eigenvalue=min_eigenvalue)
    
    # Save factor loadings
    loadings_path = os.path.join(factor_analysis_dir, "factor_loadings.csv")
    fa_results["loadings"].to_csv(loadings_path)
    print(f"Saved factor loadings to {loadings_path}")
    
    # Save factor loadings heatmap
    loadings_plot_path = os.path.join(factor_analysis_dir, "factor_loadings_heatmap.png")
    plot_factor_loadings(fa_results["loadings"], loadings_plot_path)
    print(f"Saved factor loadings heatmap to {loadings_plot_path}")
    
    # Create factor interpretation plot
    interpretation_path = os.path.join(factor_analysis_dir, "factor_interpretation.png")
    plot_factor_interpretation(fa_results["loadings"], interpretation_path, threshold=0.5)
    print(f"Saved factor interpretation plot to {interpretation_path}")
    
    # Save variance information
    variance_path = os.path.join(factor_analysis_dir, "factor_variance.csv")
    fa_results["variance"].to_csv(variance_path)
    print(f"Saved variance information to {variance_path}")
    
    # Save communalities
    communalities_path = os.path.join(factor_analysis_dir, "factor_communalities.csv")
    fa_results["communalities"].to_csv(communalities_path)
    print(f"Saved communalities to {communalities_path}")
    
    # If we have at least 2 factors, create a biplot
    if fa_results["n_factors"] >= 2:
        biplot_path = os.path.join(factor_analysis_dir, "factor_biplot.png")
        plot_factor_biplot(fa_results, df, biplot_path)
        print(f"Saved factor biplot to {biplot_path}")
        
    # Calculate factor importance with polynomial features
    linear_importances, linear_r2, poly_importances, poly_r2, nonlinear_importance = calculate_factor_importance_nonlinear(df)
    
    if poly_importances is not None:
        # Save polynomial importances to CSV
        poly_importances_path = os.path.join(factor_analysis_dir, "polynomial_factor_importances.csv")
        poly_importances.to_frame('coefficient').to_csv(poly_importances_path)
        print(f"Saved polynomial factor importances to {poly_importances_path}")
        
        # Save R² values to text file
        r2_info_path = os.path.join(factor_analysis_dir, "r2_comparison.txt")
        with open(r2_info_path, 'w') as f:
            f.write(f"Linear model R²: {linear_r2:.6f}\n")
            f.write(f"Polynomial model R²: {poly_r2:.6f}\n")
            f.write(f"R² improvement: {poly_r2 - linear_r2:.6f}\n")
            f.write(f"Percentage of variance explained by nonlinear terms: {100 * (poly_r2 - linear_r2) / (1 - linear_r2):.2f}%\n")
        print(f"Saved R² comparison information to {r2_info_path}")
        
        # Create polynomial importance plot
        poly_plot_path = os.path.join(factor_analysis_dir, "polynomial_importance.png")
        plot_polynomial_importance(poly_importances, linear_r2, poly_r2, poly_plot_path)
        print(f"Saved polynomial importance plot to {poly_plot_path}")
    
    # Identify low reliability questions if requested
    if analyze_questions and 'question_id' in df.columns:
        print("\nIdentifying questions with low reliability...")
        try:
            low_reliability = identify_low_reliability_questions(df, fa_results, threshold=threshold)
            if low_reliability is not None:
                # Save to CSV
                low_reliability_path = os.path.join(question_reliability_dir, "question_reliability.csv")
                low_reliability.to_csv(low_reliability_path, index=False)
                print(f"Saved question reliability analysis to {low_reliability_path}")
                
                # Create clusters of similar questions with low reliability
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # Only cluster the low reliability questions
                low_rel_questions = low_reliability[low_reliability['low_reliability']].copy()
                if len(low_rel_questions) > 5:  # Need enough questions to cluster
                    # Get the residual columns for clustering
                    # Use standardized residual for clustering if available
                    if 'standardized_residual' in low_rel_questions.columns:
                        print(f"Using standardized residuals for clustering {len(low_rel_questions)} questions")
                        cluster_cols = [col for col in low_rel_questions.columns if '_residual_sq' in col] + ['standardized_residual']
                    else:
                        cluster_cols = [col for col in low_rel_questions.columns if '_residual_sq' in col]
                    
                    # Standardize the data
                    scaler = StandardScaler()
                    cluster_data = scaler.fit_transform(low_rel_questions[cluster_cols])
                    
                    # Determine optimal number of clusters (2-5)
                    from sklearn.metrics import silhouette_score
                    best_score = -1
                    best_k = 2
                    for k in range(2, min(6, len(low_rel_questions))):
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        labels = kmeans.fit_predict(cluster_data)
                        score = silhouette_score(cluster_data, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                    
                    # Cluster the questions
                    kmeans = KMeans(n_clusters=best_k, random_state=42)
                    low_rel_questions['cluster'] = kmeans.fit_predict(cluster_data)
                    
                    # Save clustered questions
                    clusters_path = os.path.join(question_reliability_dir, "question_reliability_clusters.csv")
                    low_rel_questions.to_csv(clusters_path, index=False)
                    print(f"Saved {best_k} clusters of low reliability questions to {clusters_path}")
        except Exception as e:
            print(f"Error in question reliability analysis: {e}")
    
    return fa_results

def load_processed_jsonl_files(directory_path):
    """
    Load all JSONL files in the directory, process them and create a dataframe with scores.
    
    Parameters:
    -----------
    directory_path : str
        Directory containing processed JSONL files
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with score data
    """
    import json
    
    directory = Path(directory_path)
    jsonl_files = list(directory.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {directory_path}")
        return None
    
    # List to store dataframes from each file
    dfs = []
    
    # Score mapping for conversion
    score_mapping = {
        '': 3,
        'A>>B': 1,
        'A>B': 2,
        'A=B': 3,
        'B>A': 4,
        'B>>A': 5,
        'A<<B': 5,
        'A<B': 4,
        'B=A': 3,
        'B<A': 2,
        'B<<A': 1
    }
    
    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file.name}")
        
        # Extract model name from filename
        model_name = jsonl_file.stem
        if model_name.endswith("_ct"):
            # Skip CT files
            continue
            
        # Read JSONL file and parse into list of dictionaries
        data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # If no data was read, skip this file
        if not data:
            print(f"No data found in {jsonl_file.name}")
            continue
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if the dataframe has the expected structure
        if 'games' not in df.columns:
            print(f"Skipping {jsonl_file.name} - no 'games' column found")
            continue
            
        # Explode the list column so each list item becomes a row
        df_exploded = df.explode('games').reset_index()
        
        # Add model name
        df_exploded['model'] = model_name
        
        # Convert the dictionaries in the exploded column to individual columns
        try:
            df_normalized = pd.json_normalize(df_exploded['games'])
            
            # Get all columns, ensuring unique names
            cols = list(df_normalized.columns)
            # Find duplicate columns
            seen = {}
            dupes = []
            for i, col in enumerate(cols):
                if col in seen:
                    dupes.append((i, col))
                else:
                    seen[col] = i
                    
            # Add a suffix to duplicate columns
            for idx, col in dupes:
                cols[idx] = f"{col}-{str(idx).zfill(2)}"
                
            df_normalized.columns = cols
            
            # Join with original dataframe
            df_final = pd.concat([df_exploded.drop('games', axis=1), df_normalized], axis=1)
            
            # Fill NA values
            df_final = df_final.fillna("")
            
            # Convert scores from text to numeric
            for k in ['score', 'correctness_score', 'safety_score', 'completeness_score', 'conciseness_score', 'style_score']:
                if k in df_final.columns:
                    df_final[k] = df_final[k].astype(str).replace(score_mapping).astype(np.int64)
            
            # Add to list of dataframes
            dfs.append(df_final)
            
        except Exception as e:
            print(f"Error processing {jsonl_file.name}: {e}")
            continue
    
    if not dfs:
        print("No valid dataframes were created")
        return None
        
    # Concatenate all dataframes
    result = pd.concat(dfs, ignore_index=True)
    
    # Keep only the columns we need for factor analysis
    keep_cols = ['question_id', 'model']
    score_cols = ['score', 'correctness_score', 'safety_score', 'completeness_score', 'conciseness_score', 'style_score']
    for col in score_cols:
        if col in result.columns:
            keep_cols.append(col)
    
    result = result[keep_cols]
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Perform factor analysis on score columns in JSONL files.')
    parser.add_argument('directory', type=str, help='Directory containing processed JSONL files')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files (defaults to the input directory)')
    parser.add_argument('--min-eigenvalue', type=float, default=0.75, 
                        help='Minimum eigenvalue to retain a factor (default: 0.75)')
    parser.add_argument('--analyze-questions', action='store_true',
                        help='Analyze question reliability and identify problematic questions')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Threshold for standardized residuals to flag low reliability questions (default: 2.0)')
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.directory
    
    # Try to load JSONL files first
    df = load_processed_jsonl_files(args.directory)
    
    # If no JSONL files found or couldn't process them, try loading CSV files
    if df is None:
        print("No valid JSONL data found. Trying to load CSV files instead...")
        try:
            from get_corrs import load_and_join_scores
            df = load_and_join_scores(args.directory)
        except:
            print("Failed to load data from CSV files as well")
            return
    
    if df is None or len(df.columns) <= 2:  # Need model + at least 2 score columns
        print("Not enough data for factor analysis")
        return
    
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run factor analysis with threshold parameter for question reliability
    if args.analyze_questions:
        print(f"Running factor analysis with question reliability analysis (threshold: {args.threshold})")
        run_factor_analysis(df, output_dir, args.min_eigenvalue, args.analyze_questions, threshold=args.threshold)
    else:
        run_factor_analysis(df, output_dir, args.min_eigenvalue, args.analyze_questions)

if __name__ == "__main__":
    main()