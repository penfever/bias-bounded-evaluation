import pandas as pd
import numpy as np
import plotly.express as px

import datetime
import argparse
import os
import math

from glob import glob
from tqdm import tqdm
import inspect

from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from utils import load_model_answers

# Ensure compatibility with numpy 2.0+
if not hasattr(np, 'NAN'):
    np.NAN = np.nan

# For Bayesian bootstrap
try:
    import scipy.stats as stats
except ImportError:
    print("Warning: scipy not found, Bayesian bootstrap will not be available")
    stats = None


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, baseline_model="gpt-4-0314"):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt-4-0314 = 1000
    if baseline_model in models.index:
        elo_scores += 1000 - elo_scores[models[baseline_model]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round, baseline_model="gpt-4-0314", bootstrap_method="standard"):
    """
    Perform bootstrap analysis of model ELO ratings with different bootstrapping methods.
    
    Parameters:
    -----------
    battles : pandas.DataFrame
        DataFrame containing battle data
    func_compute_elo : function
        Function to compute ELO ratings
    num_round : int
        Number of bootstrap rounds
    baseline_model : str, optional
        Baseline model for ELO calculation
    bootstrap_method : str, optional
        Bootstrapping method to use: "standard" or "bayesian"
        
    Returns:
    --------
    pandas.DataFrame : Bootstrap results
    """
    # Standard bootstrap (original implementation)
    if bootstrap_method == "standard":
        rows = []
        kwargs = {}
        if baseline_model in inspect.signature(func_compute_elo).parameters:
            kwargs[baseline_model] = baseline_model
        for _ in tqdm(range(num_round), desc="standard bootstrap"):
            rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True), **kwargs))
        df = pd.DataFrame(rows)
        return df[df.median().sort_values(ascending=False).index]
    
    # Bayesian hierarchical bootstrap
    elif bootstrap_method == "bayesian":
        # Ensure we have scipy.stats available
        if stats is None:
            print("Error: scipy.stats module is required for Bayesian bootstrap")
            print("Falling back to standard bootstrap")
            return get_bootstrap_result(battles, func_compute_elo, num_round, baseline_model, "standard")
        
        print(f"Using Bayesian hierarchical bootstrap")
        
        # Standard bootstrap to get base distribution (smaller sample for efficiency)
        print("Generating initial ELO distribution...")
        base_rounds = min(100, num_round // 2)  # Use fewer rounds for initial distribution
        base_rows = []
        kwargs = {}
        if baseline_model in inspect.signature(func_compute_elo).parameters:
            kwargs[baseline_model] = baseline_model
            
        for _ in tqdm(range(base_rounds), desc="initial bootstrap"):
            sampled_battles = battles.sample(frac=1.0, replace=True)
            base_rows.append(func_compute_elo(sampled_battles, **kwargs))
        
        base_df = pd.DataFrame(base_rows)
        
        # For each model, create a hierarchical model with appropriate variance
        print("Performing Bayesian hierarchical sampling...")
        final_rows = []
        for _ in tqdm(range(num_round), desc="bayesian bootstrap"):
            sample_row = {}
            
            for model in base_df.columns:
                # Get distribution parameters
                mu = base_df[model].mean()
                sigma = base_df[model].std()
                
                # Skip if sigma is too small (can happen with baseline model)
                if sigma < 1e-6:
                    sample_row[model] = mu
                    continue
                
                # Sample from the t-distribution (more robust than normal)
                # Use 5 degrees of freedom for slightly heavier tails
                sample_row[model] = stats.t.rvs(df=5, loc=mu, scale=sigma)
                
            final_rows.append(pd.Series(sample_row))
            
        df = pd.DataFrame(final_rows)
        return df[df.median().sort_values(ascending=False).index]
    
    # Fallback for any other method
    else:
        print(f"Bootstrap method '{bootstrap_method}' not recognized. Using standard bootstrap.")
        return get_bootstrap_result(battles, func_compute_elo, num_round, baseline_model, "standard")


def preety_print_two_ratings(ratings_1, ratings_2, column_names):
    df = pd.DataFrame([
        [n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()
    ], columns=["Model", column_names[0], column_names[1]]).sort_values(column_names[0], ascending=False).reset_index(drop=True)
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus", text="rating_rounded",
                     title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating",
                      height=600)
    return fig


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.nan for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(df, column, baseline="gpt-4-0314"):
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_battles_from_judgment(judge_name, first_game_only=False, WEIGHT=3, baseline_model="gpt-4-0314", args=None):
    arena_hard_battles = pd.DataFrame()
    
    print("Turning judgment results into battles...")
    if args.judgment_dir == "":
        directory = f"data/arena-hard-v0.1/model_judgment/{judge_name}_judge/{baseline_model}_base"
    else:
        directory = args.judgment_dir
        
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        # Return empty DataFrame with the expected columns
        return pd.DataFrame(columns=["question_id", "model_a", "model_b", "winner"])
    
    # Check if directory contains any JSONL files
    jsonl_files = glob(f"{directory}/*jsonl")
    if not jsonl_files:
        print(f"Warning: No JSONL files found in {directory}")
        # Return empty DataFrame with the expected columns
        return pd.DataFrame(columns=["question_id", "model_a", "model_b", "winner"])
    
    # Process each JSONL file
    for file in tqdm(jsonl_files):
        try:
            df = pd.read_json(file, lines=True)
            
            # Check if DataFrame is empty
            if df.empty:
                print(f"Warning: Empty file: {file}")
                continue
                
            # Check if required columns exist
            if "question_id" not in df.columns or "model" not in df.columns or "games" not in df.columns:
                print(f"Warning: File {file} missing required columns")
                continue
                
            for _, row in df.iterrows():
                try:
                    # Check if games is a list and has at least one element
                    if not isinstance(row["games"], list) or len(row["games"]) == 0:
                        print(f"Warning: Invalid games data in file {file}")
                        continue
                    
                    # game 1
                    output = {"question_id": row["question_id"],
                            "model_a": baseline_model,
                            "model_b": row["model"]}

                    # Check if the target metric exists in the game data
                    game = row["games"][0]
                    if args.target_metric not in game:
                        print(f"Warning: Metric {args.target_metric} not found in game data")
                        continue

                    weight = 1
                    if game[args.target_metric] == "A=B":
                        output["winner"] = "tie"
                    elif game[args.target_metric] == "A>B":
                        output["winner"] = "model_a"
                    elif game[args.target_metric] == "A>>B":
                        output["winner"] = "model_a"
                        weight = WEIGHT
                    elif game[args.target_metric] == "B>A":
                        output["winner"] = "model_b"
                    elif game[args.target_metric] == "B>>A":
                        output["winner"] = "model_b"
                        weight = WEIGHT
                    else:
                        weight = 0

                    if weight:
                        arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

                    if not first_game_only and len(row["games"]) > 1:
                        # game 2
                        output = {"question_id": row["question_id"],
                                "model_a": baseline_model,
                                "model_b": row["model"]}

                        game = row["games"][1]
                        
                        # Check if the target metric exists in the second game
                        if args.target_metric not in game:
                            continue

                        weight = 1
                        if game[args.target_metric] == "A=B":
                            output["winner"] = "tie"
                        elif game[args.target_metric] == "A>B":
                            output["winner"] = "model_b"
                        elif game[args.target_metric] == "A>>B":
                            output["winner"] = "model_b"
                            weight = WEIGHT
                        elif game[args.target_metric] == "B>A":
                            output["winner"] = "model_a"
                        elif game[args.target_metric] == "B>>A":
                            output["winner"] = "model_a"
                            weight = WEIGHT
                        else:
                            weight = 0

                        if weight:
                            arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    
    # Check if we found any valid battles
    if arena_hard_battles.empty:
        print("Warning: No valid battles found")
        return pd.DataFrame(columns=["question_id", "model_a", "model_b", "winner"])
    
    # Save battles to file
    os.makedirs("data", exist_ok=True)
    arena_hard_battles.to_json("data/arena_hard_battles.jsonl", lines=True, orient="records")
    return arena_hard_battles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena-hard-v0.1")
    parser.add_argument("--judge-name", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--baseline", type=str, default="gpt-4-0314")
    parser.add_argument("--load-battles", action="store_true")
    parser.add_argument("--load-bootstrap", action="store_true")
    parser.add_argument("--show-elo", action="store_true")
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--first-game-only", action="store_true")
    parser.add_argument("--answer-dir", type=str, default="")
    parser.add_argument("--judgment-dir", type=str, default="")
    parser.add_argument("--target-metric", type=str, default="score")
    parser.add_argument("--bootstrap-method", type=str, choices=['standard', 'bayesian'], 
                        default='standard', help='Bootstrapping method to use')
    parser.add_argument("--communalities-file", type=str, default="", 
                        help='Path to factor analysis communalities file to adjust bootstrap confidence intervals')
    parser.add_argument("--reliability-file", type=str, default="", 
                        help='Path to reliability metrics file to further adjust bootstrap confidence intervals')
    args = parser.parse_args()
    print(args)
    assert not args.load_bootstrap or (args.load_battles and args.load_bootstrap), "If loading prexisting bootstrapping data, you must also load preexisting battles."
    if args.answer_dir == "":
        answer_dir = os.path.join("data", args.bench_name, "model_answer")
    else:
        answer_dir = args.answer_dir
    model_answers = load_model_answers(answer_dir)
    
    if args.load_battles:
        assert os.path.exists("data/arena_hard_battles.jsonl")
        battles = pd.read_json("data/arena_hard_battles.jsonl", lines=True)
    else:
        battles = get_battles_from_judgment(args.judge_name, args.first_game_only, args.weight, args.baseline, args)
        
    bootstrap_online_elo = compute_mle_elo(battles, baseline_model=args.baseline)


    if args.load_bootstrap:
        bootstrap_elo_lu = pd.read_json("data/bootstrapping_results.jsonl", lines=True)
    else:
        np.random.seed(42)
        bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, args.num_rounds, args.baseline, args.bootstrap_method)
        bootstrap_elo_lu.to_json("data/bootstrapping_results.jsonl", lines=True, orient="records")
        
    # Initialize variance adjustment factors
    adjustment_factors = []
    
    # Load communalities file if provided to adjust confidence intervals
    if args.communalities_file and os.path.exists(args.communalities_file):
        try:
            print(f"Loading communalities file: {args.communalities_file}")
            communalities_df = pd.read_csv(args.communalities_file)
            # First column might be named differently
            first_col = communalities_df.columns[0]
            metric_col = f"{args.target_metric}"
            
            # For general "score" metric, use mean communality of all factors
            if metric_col == "score":
                # Calculate mean of all communalities
                mean_communality = communalities_df['Communality'].mean()
                communality = mean_communality
                print(f"Using mean communality ({mean_communality:.4f}) for general 'score' metric")
            # Otherwise, check if the specific metric exists in the communalities file
            elif metric_col in communalities_df[first_col].values:
                communality = communalities_df.loc[communalities_df[first_col] == metric_col, 'Communality'].values[0]
            else:
                print(f"Warning: Target metric '{metric_col}' not found in communalities file")
                communality = 1.0
                
            # Calculate adjustment factor based on communality
            # Using variance components model: sqrt(var1 + var2)
            # var1 is the bootstrap variance, var2 is the residual variance (1 - communality)
            if communality > 0:
                communality_factor = np.sqrt(1 / communality)
                print(f"Adjusting confidence intervals using communality factor: {communality_factor:.4f}")
                adjustment_factors.append((1 - communality) / communality)  # Store variance component for later combination
            else:
                print("Warning: Invalid communality value (zero or negative). Not using this adjustment.")
        except Exception as e:
            print(f"Error loading communalities file: {e}")
    
    # Load reliability metrics file if provided to further adjust confidence intervals
    if args.reliability_file and os.path.exists(args.reliability_file):
        try:
            print(f"Loading reliability metrics file: {args.reliability_file}")
            reliability_df = pd.read_csv(args.reliability_file)
            # First column might be named differently
            first_col = reliability_df.columns[0]
            metric_col = f"{args.target_metric}"
            
            # For general "score" metric, use mean reliability of all factors
            if metric_col == "score":
                # Calculate mean of all reliability scores
                mean_reliability = reliability_df['reliability_score'].mean()
                reliability = mean_reliability
                print(f"Using mean reliability ({mean_reliability:.4f}) for general 'score' metric")
            # Otherwise, check if the specific metric exists in the reliability file
            elif metric_col in reliability_df[first_col].values:
                reliability = reliability_df.loc[reliability_df[first_col] == metric_col, 'reliability_score'].values[0]
            else:
                print(f"Warning: Target metric '{metric_col}' not found in reliability file")
                reliability = 1.0
                
            # Calculate adjustment factor based on reliability
            # Using the same variance components model approach
            if reliability > 0 and reliability < 1:  # reliability should be between 0 and 1
                reliability_factor = np.sqrt(1 / reliability)
                print(f"Adjusting confidence intervals using reliability factor: {reliability_factor:.4f}")
                adjustment_factors.append((1 - reliability) / reliability)  # Store variance component for later combination
            else:
                print("Warning: Invalid reliability value. Not using this adjustment.")
        except Exception as e:
            print(f"Error loading reliability file: {e}")
    
    # Combine all variance components using the variance components model
    # sqrt(var1 + var2 + var3 + ...)
    if adjustment_factors:
        # Calculate combined variance components
        combined_variance_ratio = sum(adjustment_factors)
        # Convert to a scaling factor for confidence intervals
        combined_factor = np.sqrt(1 + combined_variance_ratio)
        print(f"Combined adjustment factor from {len(adjustment_factors)} sources: {combined_factor:.4f}")
    else:
        combined_factor = 1.0
        print("No valid adjustment factors found. Using standard confidence intervals.")

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats['results'].astype('object')

    for i, model in enumerate(bootstrap_online_elo.index):
        assert model in bootstrap_elo_lu.columns

        stats.at[i, "model"] = model
        stats.at[i, args.target_metric] = bootstrap_online_elo[model]
        
        # Calculate adjusted confidence intervals using combined factor
        if combined_factor > 1.0:
            # Get the median value
            median = np.median(bootstrap_elo_lu[model])
            # Calculate the distance from the median
            lower_distance = median - np.percentile(bootstrap_elo_lu[model], 2.5)
            upper_distance = np.percentile(bootstrap_elo_lu[model], 97.5) - median
            # Apply the combined factor to adjust the distances
            adjusted_lower = median - (lower_distance * combined_factor)
            adjusted_upper = median + (upper_distance * combined_factor)
            stats.at[i, "lower"] = adjusted_lower
            stats.at[i, "upper"] = adjusted_upper
        else:
            # Use standard percentiles if no adjustment needed
            stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
            stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)

        length = 0
        if model in model_answers:
            for _, row in model_answers[model].items():
                turn = row["choices"][0]["turns"][0]
                length += turn["token_len"]
            length /= len(model_answers[model])

        stats.at[i, "avg_tokens"] = int(length)
        stats.at[i, "results"] = bootstrap_elo_lu[model].tolist()
    
    if not args.show_elo:
        stats.sort_values(by="model", inplace=True)
        stats[args.target_metric] = get_win_rate_column(stats, args.target_metric, args.baseline).tolist()
        stats["lower"] = get_win_rate_column(stats, "lower", args.baseline).tolist()
        stats["upper"] = get_win_rate_column(stats, "upper", args.baseline).tolist()
        decimal = 1
    else:
        decimal = 0
        stats = stats.astype({args.target_metric : int, "lower" : int, "upper" : int})
    
    stats.sort_values(by=args.target_metric, ascending=False, inplace=True)
    for _, row in stats.iterrows():
        interval = str((round(row['lower'] - row[args.target_metric], decimal), round(row['upper'] - row[args.target_metric], decimal)))
        print(f"{row['model'] : <30} | score: {round(row[args.target_metric], decimal) : ^5} | 95% CI: {interval : ^12} | average #tokens: {int(row['avg_tokens'])}")

    if args.output:
        cur_date = datetime.datetime.now()
        date_str = cur_date.strftime("%Y%m%d")
        stats = stats.drop(columns=['results'])
        CI = []
        for i in range(len(stats)):
            score = stats.iloc[i][args.target_metric]
            upper = stats.iloc[i]['upper']
            lower = stats.iloc[i]['lower']
            CI.append(f"(-{(score-lower):.2f}, +{(upper-score):.2f})")

        stats["CI"] = CI
        col_list = list(stats)
        stats = stats.loc[:,col_list]
        stats.rename(columns={'upper': 'rating_q975'}, inplace=True)
        stats.rename(columns={'lower': 'rating_q025'}, inplace=True)

        col_list = list(stats)
        col_list[-2], col_list[-1] = col_list[-1], col_list[-2]
        stats = stats.loc[:,col_list]
        stats['date'] = date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:]
        # Add indicators to filename based on what adjustment factors were used
        adjustment_suffix = ""
        if args.communalities_file and args.reliability_file:
            adjustment_suffix = "_variance_combined"
        elif args.communalities_file:
            adjustment_suffix = "_communality"
        elif args.reliability_file:
            adjustment_suffix = "_reliability"
            
        # Create leaderboard directory if it doesn't exist
        os.makedirs("leaderboard", exist_ok=True)
        
        # Determine the output directory based on adjustment factors
        if args.communalities_file or args.reliability_file:
            output_subdir = "factor_scores_updated_cis"
        else:
            output_subdir = "factor_scores_original_cis"
            
        # Create the filename
        filename = f"arena_hard_leaderboard_{date_str}_{args.judge_name}_judge_{args.baseline}_base_{args.target_metric}_factor{adjustment_suffix}.csv"
        
        # Save to the leaderboard directory (will be moved by the shell script)
        stats.to_csv(f"leaderboard/{filename}", index=False)