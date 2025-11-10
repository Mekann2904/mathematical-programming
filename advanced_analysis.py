"""
高度な分析と統計的評価 - ナップサック問題のアルゴリズム比較
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from greedy_knapsack import (
    greedy_value_density, greedy_value_only, greedy_weight_only,
    greedy_hybrid, improved_greedy, knapsack_dp
)
import time
import scipy.stats as stats

def generate_random_instances(num_instances: int, n: int, capacity_range: Tuple[int, int] = (50, 200)) -> List[Tuple]:
    """ランダムな問題インスタンスを生成"""
    instances = []
    for i in range(num_instances):
        # ランダムなパラメータ生成
        capacity = np.random.randint(capacity_range[0], capacity_range[1])
        weights = np.random.randint(1, 20, n)
        values = np.random.randint(1, 50, n)
        instances.append((weights.tolist(), values.tolist(), capacity, f"random_{i+1}"))
    return instances

def run_statistical_analysis(num_trials: int = 100, problem_size: int = 20):
    """統計的分析の実行"""
    print("統計的分析を実行中...")
    
    # ランダムインスタンスの生成
    instances = generate_random_instances(num_trials, problem_size)
    
    # アルゴリズムの定義
    algorithms = [
        ("greedy_value_density", greedy_value_density),
        ("greedy_value_only", greedy_value_only),
        ("greedy_weight_only", greedy_weight_only),
        ("greedy_hybrid", greedy_hybrid),
        ("improved_greedy", improved_greedy),
        ("knapsack_dp", knapsack_dp)
    ]
    
    results = []
    
    for instance_idx, (weights, values, capacity, name) in enumerate(instances):
        # 最適解の計算（動的計画法）
        try:
            opt_value, _, _, _ = knapsack_dp(weights, values, capacity)
        except:
            continue
        
        for algo_name, algo_func in algorithms:
            try:
                if algo_name == "knapsack_dp":
                    value, _, _, exec_time = algo_func(weights, values, capacity)
                else:
                    _, _, value, exec_time = algo_func(weights, values, capacity)
                
                # 近似比の計算
                if opt_value > 0:
                    approximation_ratio = value / opt_value
                else:
                    approximation_ratio = 1.0
                
                results.append({
                    'instance': name,
                    'algorithm': algo_name,
                    'value': value,
                    'optimal_value': opt_value,
                    'approximation_ratio': approximation_ratio,
                    'execution_time': exec_time,
                    'problem_size': problem_size
                })
            except Exception as e:
                print(f"エラー: {algo_name} on {name}: {e}")
                continue
    
    return pd.DataFrame(results)

def analyze_approximation_ratios(df: pd.DataFrame):
    """近似比の詳細分析"""
    print("\n近似比の統計的分析:")
    print("=" * 50)
    
    # 基本統計量
    stats_summary = df.groupby('algorithm')['approximation_ratio'].agg([
        'count', 'mean', 'std', 'min', 'max', 
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 50),
        lambda x: np.percentile(x, 75)
    ]).round(4)
    
    stats_summary.columns = ['count', 'mean', 'std', 'min', 'max', '25%', '50%', '75%']
    print(stats_summary)
    
    return stats_summary

def analyze_execution_times(df: pd.DataFrame):
    """実行時間の詳細分析"""
    print("\n実行時間の統計的分析 (秒):")
    print("=" * 50)
    
    time_stats = df.groupby('algorithm')['execution_time'].agg([
        'mean', 'std', 'min', 'max'
    ]).round(6)
    
    print(time_stats)
    
    return time_stats

def perform_statistical_tests(df: pd.DataFrame):
    """統計的検定の実行"""
    print("\n統計的検定:")
    print("=" * 50)
    
    algorithms = df['algorithm'].unique()
    
    # 近似比の正規性検定
    print("近似比の正規性検定 (Shapiro-Wilk):")
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]['approximation_ratio']
        stat, p_value = stats.shapiro(algo_data)
        print(f"{algo:20} W={stat:.4f}, p={p_value:.4f} {'(正規分布)' if p_value > 0.05 else '(非正規分布)'}")
    
    print("\nアルゴリズム間の近似比比較 (Kruskal-Wallis検定):")
    groups = [df[df['algorithm'] == algo]['approximation_ratio'] for algo in algorithms]
    h_stat, p_value = stats.kruskal(*groups)
    print(f"H統計量={h_stat:.4f}, p値={p_value:.4f}")
    if p_value < 0.05:
        print("アルゴリズム間に統計的有意差あり (p < 0.05)")
    else:
        print("アルゴリズム間に統計的有意差なし")
    
    # 多重比較 (Mann-Whitney U検定)
    print("\nアルゴリズム間の多重比較 (Mann-Whitney U検定):")
    comparisons = []
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            algo1_data = df[df['algorithm'] == algorithms[i]]['approximation_ratio']
            algo2_data = df[df['algorithm'] == algorithms[j]]['approximation_ratio']
            u_stat, p_value = stats.mannwhitneyu(algo1_data, algo2_data, alternative='two-sided')
            comparisons.append({
                'algorithm1': algorithms[i],
                'algorithm2': algorithms[j],
                'u_stat': u_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    comparisons_df = pd.DataFrame(comparisons)
    print(comparisons_df.round(4))
    
    return comparisons_df

def create_comprehensive_visualizations(df: pd.DataFrame, stats_summary: pd.DataFrame):
    """包括的な可視化の作成"""
    print("\n包括的な可視化を作成中...")
    
    # Use default English font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 近似比の分布（ボックスプロット）
    sns.boxplot(data=df, x='algorithm', y='approximation_ratio', ax=axes[0,0])
    axes[0,0].set_title('Approximation Ratio Distribution by Algorithm')
    axes[0,0].set_xticks(range(len(method_names)))
    axes[0,0].set_xticklabels(method_names, rotation=45)
    axes[0,0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Optimal')
    
    # 2. 実行時間の分布（ボックスプロット）
    sns.boxplot(data=df, x='algorithm', y='execution_time', ax=axes[0,1])
    axes[0,1].set_title('Execution Time Distribution by Algorithm')
    axes[0,1].set_xticks(range(len(method_names)))
    axes[0,1].set_xticklabels(method_names, rotation=45)
    axes[0,1].set_yscale('log')  # Log scale
    
    # 3. 近似比のヒストグラム
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]['approximation_ratio']
        axes[0,2].hist(algo_data, alpha=0.6, label=algo, bins=20)
    axes[0,2].set_title('Approximation Ratio Histogram')
    axes[0,2].legend()
    axes[0,2].axvline(x=1.0, color='r', linestyle='--', alpha=0.7)
    
    # 4. アルゴリズム性能のヒートマップ（平均近似比）
    pivot_table = df.pivot_table(values='approximation_ratio', index='instance', columns='algorithm', aggfunc='mean')
    sns.heatmap(pivot_table, ax=axes[1,0], cmap='YlOrRd', cbar_kws={'label': 'Approximation Ratio'})
    axes[1,0].set_title('Algorithm Performance Heatmap')
    
    # 5. 実行時間 vs 近似比（散布図）
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        axes[1,1].scatter(algo_data['execution_time'], algo_data['approximation_ratio'], 
                         alpha=0.6, label=algo, s=30)
    axes[1,1].set_xlabel('Execution Time (s)')
    axes[1,1].set_ylabel('Approximation Ratio')
    axes[1,1].set_title('Time vs Quality Trade-off')
    axes[1,1].set_xscale('log')
    axes[1,1].legend()
    axes[1,1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    # 6. アルゴリズムランキング
    algo_ranking = stats_summary.sort_values('mean', ascending=False)
    axes[1,2].barh(range(len(algo_ranking)), algo_ranking['mean'])
    axes[1,2].set_yticks(range(len(algo_ranking)))
    axes[1,2].set_yticklabels(algo_ranking.index)
    axes[1,2].set_xlabel('Mean Approximation Ratio')
    axes[1,2].set_title('Algorithm Ranking by Mean Performance')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_worst_case_performance(df: pd.DataFrame):
    """最悪ケース性能の分析"""
    print("\n最悪ケース性能分析:")
    print("=" * 50)
    
    # 各アルゴリズムの最低近似比
    worst_cases = df.groupby('algorithm')['approximation_ratio'].min().sort_values()
    
    print("最低近似比:")
    for algo, ratio in worst_cases.items():
        print(f"{algo:25}: {ratio:.4f}")
    
    # 近似比が0.5未満のインスタンス数
    poor_performance = df[df['approximation_ratio'] < 0.5].groupby('algorithm').size()
    print(f"\n近似比0.5未満のインスタンス数:")
    for algo, count in poor_performance.items():
        print(f"{algo:25}: {count} instances")
    
    return worst_cases, poor_performance

def main():
    """メイン分析関数"""
    print("高度な統計的分析を開始します...")
    
    # 統計的分析の実行
    df = run_statistical_analysis(num_trials=50, problem_size=20)
    
    # 各種分析の実行
    stats_summary = analyze_approximation_ratios(df)
    time_stats = analyze_execution_times(df)
    comparisons = perform_statistical_tests(df)
    worst_cases, poor_performance = analyze_worst_case_performance(df)
    
    # 可視化の作成
    create_comprehensive_visualizations(df, stats_summary)
    
    # 結果の保存
    results = {
        'summary_stats': stats_summary,
        'time_stats': time_stats,
        'statistical_tests': comparisons,
        'worst_cases': worst_cases,
        'poor_performance': poor_performance
    }
    
    # CSVとして保存
    df.to_csv('statistical_analysis_results.csv', index=False)
    stats_summary.to_csv('approximation_stats.csv')
    
    print("\n分析完了！結果を保存しました:")
    print("- statistical_analysis_results.csv")
    print("- approximation_stats.csv")
    print("- comprehensive_analysis.png")
    
    return results

if __name__ == "__main__":
    results = main()