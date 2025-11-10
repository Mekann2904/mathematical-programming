"""
大規模問題評価 - ナップサック問題のアルゴリズム性能評価
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from typing import List, Tuple, Dict
from greedy_knapsack import (
    greedy_value_density, greedy_value_only, greedy_weight_only,
    greedy_hybrid, improved_greedy, knapsack_dp
)

def generate_large_instances(sizes: List[int], num_instances: int = 5) -> Dict[int, List[Tuple]]:
    """大規模な問題インスタンスを生成"""
    instances = {}
    for size in sizes:
        size_instances = []
        for i in range(num_instances):
            # 問題サイズに応じたパラメータ設定
            capacity = size * 10  # 容量は品物数に比例
            weights = np.random.randint(1, 50, size)
            values = np.random.randint(1, 100, size)
            size_instances.append((weights.tolist(), values.tolist(), capacity, f"large_{size}_{i+1}"))
        instances[size] = size_instances
    return instances

def evaluate_large_scale_performance(max_size: int = 200, step: int = 20):
    """大規模問題での性能評価"""
    print("大規模問題の性能評価を実行中...")
    
    # 評価する問題サイズ
    sizes = list(range(20, max_size + 1, step))
    
    # アルゴリズムリスト（動的計画法は小規模問題のみ）
    algorithms = [
        ("greedy_value_density", greedy_value_density),
        ("greedy_value_only", greedy_value_only), 
        ("greedy_weight_only", greedy_weight_only),
        ("greedy_hybrid", greedy_hybrid),
        ("improved_greedy", improved_greedy),
    ]
    
    results = []
    
    for size in sizes:
        print(f"問題サイズ {size} を評価中...")
        
        # ランダムインスタンス生成
        capacity = size * 10
        weights = np.random.randint(1, 50, size)
        values = np.random.randint(1, 100, size)
        
        # 動的計画法は小規模問題のみ実行
        if size <= 100:
            try:
                start_time = time.perf_counter()
                opt_value, _, _, _ = knapsack_dp(weights.tolist(), values.tolist(), capacity)
                dp_time = time.perf_counter() - start_time
                
                results.append({
                    'problem_size': size,
                    'algorithm': 'knapsack_dp',
                    'execution_time': dp_time,
                    'memory_usage': size * capacity,  # DPテーブルサイズ
                    'optimal_value': opt_value,
                    'feasible': True
                })
            except MemoryError:
                print(f"動的計画法: メモリ不足 (サイズ={size})")
                results.append({
                    'problem_size': size,
                    'algorithm': 'knapsack_dp', 
                    'execution_time': float('inf'),
                    'memory_usage': size * capacity,
                    'optimal_value': None,
                    'feasible': False
                })
        
        # 欲張り法の評価
        for algo_name, algo_func in algorithms:
            try:
                start_time = time.perf_counter()
                if algo_name == "knapsack_dp":
                    value, _, _, exec_time = algo_func(weights.tolist(), values.tolist(), capacity)
                else:
                    _, _, value, exec_time = algo_func(weights.tolist(), values.tolist(), capacity)
                
                # 近似比の計算（最適解が利用可能な場合）
                approximation_ratio = None
                if size <= 100 and 'optimal_value' in locals() and opt_value > 0:
                    approximation_ratio = value / opt_value
                
                results.append({
                    'problem_size': size,
                    'algorithm': algo_name,
                    'execution_time': exec_time,
                    'memory_usage': size,  # ソートのメモリ使用量
                    'value': value,
                    'approximation_ratio': approximation_ratio,
                    'feasible': True
                })
                
            except Exception as e:
                print(f"エラー: {algo_name} (サイズ={size}): {e}")
                results.append({
                    'problem_size': size,
                    'algorithm': algo_name,
                    'execution_time': float('inf'),
                    'memory_usage': size,
                    'value': None,
                    'approximation_ratio': None,
                    'feasible': False
                })
    
    return pd.DataFrame(results)

def analyze_scalability_trends(df: pd.DataFrame):
    """スケーラビリティ傾向の分析"""
    print("\nスケーラビリティ傾向分析:")
    print("=" * 50)
    
    # 実行可能なアルゴリズムのみフィルタリング
    feasible_df = df[df['feasible'] == True]
    
    # 問題サイズごとの平均実行時間
    scalability_stats = feasible_df.groupby(['problem_size', 'algorithm'])['execution_time'].agg([
        'mean', 'std'
    ]).reset_index()
    
    print("スケーラビリティ統計:")
    print(scalability_stats.round(6))
    
    return scalability_stats

def plot_scalability_results(df: pd.DataFrame, scalability_stats: pd.DataFrame):
    """スケーラビリティ結果の可視化"""
    print("\nスケーラビリティ結果を可視化中...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # アルゴリズムごとの色設定
    algorithms = df['algorithm'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    # 1. 実行時間のスケーラビリティ
    for i, algo in enumerate(algorithms):
        algo_data = scalability_stats[scalability_stats['algorithm'] == algo]
        if not algo_data.empty:
            ax1.plot(algo_data['problem_size'], algo_data['mean'], 
                    'o-', label=algo, color=colors[i], linewidth=2)
            # 標準偏差の表示
            ax1.fill_between(algo_data['problem_size'],
                           algo_data['mean'] - algo_data['std'],
                           algo_data['mean'] + algo_data['std'],
                           alpha=0.2, color=colors[i])
    
    ax1.set_xlabel('Problem Size (Number of Items)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Algorithm Scalability: Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 対数スケール
    
    # 2. メモリ使用量の比較
    memory_data = df[df['feasible'] == True].groupby(['problem_size', 'algorithm'])['memory_usage'].mean().reset_index()
    for algo in algorithms:
        algo_memory = memory_data[memory_data['algorithm'] == algo]
        if not algo_memory.empty:
            ax2.plot(algo_memory['problem_size'], algo_memory['memory_usage'], 
                    's-', label=algo, linewidth=2)
    
    ax2.set_xlabel('Problem Size (Number of Items)')
    ax2.set_ylabel('Memory Usage (estimated)')
    ax2.set_title('Algorithm Scalability: Memory Usage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. 近似比の推移（小規模問題のみ）
    small_problems = df[(df['problem_size'] <= 100) & (df['approximation_ratio'].notna())]
    if not small_problems.empty:
        approximation_data = small_problems.groupby(['problem_size', 'algorithm'])['approximation_ratio'].mean().reset_index()
        for algo in algorithms:
            algo_approx = approximation_data[approximation_data['algorithm'] == algo]
            if not algo_approx.empty:
                ax3.plot(algo_approx['problem_size'], algo_approx['approximation_ratio'], 
                        '^-', label=algo, linewidth=2)
        
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Optimal')
        ax3.set_xlabel('Problem Size (Number of Items)')
        ax3.set_ylabel('Approximation Ratio')
        ax3.set_title('Approximation Ratio vs Problem Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. アルゴリズムの限界サイズ
    max_sizes = df[df['feasible'] == True].groupby('algorithm')['problem_size'].max()
    algorithms_ordered = max_sizes.sort_values(ascending=False).index
    
    ax4.barh(range(len(algorithms_ordered)), max_sizes[algorithms_ordered])
    ax4.set_yticks(range(len(algorithms_ordered)))
    ax4.set_yticklabels(algorithms_ordered)
    ax4.set_xlabel('Maximum Feasible Problem Size')
    ax4.set_title('Algorithm Limits: Maximum Problem Size')
    
    plt.tight_layout()
    plt.savefig('large_scale_scalability.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_computational_complexity(df: pd.DataFrame):
    """計算量の実証的分析"""
    print("\n計算量の実証的分析:")
    print("=" * 50)
    
    # 実行時間の増加傾向を分析
    complexity_results = []
    
    for algo in df['algorithm'].unique():
        algo_data = df[(df['algorithm'] == algo) & (df['feasible'] == True)]
        if len(algo_data) < 3:
            continue
            
        sizes = algo_data['problem_size'].values
        times = algo_data['execution_time'].values
        
        # 対数変換して線形回帰
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        
        # 線形回帰で傾き（指数）を推定
        slope, intercept = np.polyfit(log_sizes, log_times, 1)
        
        # 推定された計算量
        estimated_complexity = f"O(n^{slope:.2f})"
        
        complexity_results.append({
            'algorithm': algo,
            'estimated_complexity': estimated_complexity,
            'slope': slope,
            'r_squared': np.corrcoef(log_sizes, log_times)[0,1]**2
        })
    
    complexity_df = pd.DataFrame(complexity_results)
    print(complexity_df.round(4))
    
    return complexity_df

def perform_memory_analysis():
    """メモリ使用量の分析"""
    print("\nメモリ使用量分析:")
    print("=" * 50)
    
    # 理論的なメモリ使用量
    memory_requirements = {
        'greedy_value_density': 'O(n) - ソートと選択',
        'greedy_value_only': 'O(n) - ソートと選択', 
        'greedy_weight_only': 'O(n) - ソートと選択',
        'greedy_hybrid': 'O(n) - 複数戦略の組み合わせ',
        'improved_greedy': 'O(n²) - 局所探索を含む',
        'knapsack_dp': 'O(nW) - DPテーブル'
    }
    
    for algo, requirement in memory_requirements.items():
        print(f"{algo:20}: {requirement}")
    
    return memory_requirements

def main():
    """メイン評価関数"""
    print("大規模問題評価を開始します...")
    
    # 大規模問題の性能評価
    df = evaluate_large_scale_performance(max_size=200, step=20)
    
    # 各種分析の実行
    scalability_stats = analyze_scalability_trends(df)
    complexity_df = analyze_computational_complexity(df)
    memory_requirements = perform_memory_analysis()
    
    # 可視化
    plot_scalability_results(df, scalability_stats)
    
    # 結果の保存
    df.to_csv('large_scale_results.csv', index=False)
    scalability_stats.to_csv('scalability_stats.csv', index=False)
    complexity_df.to_csv('computational_complexity.csv', index=False)
    
    print("\n大規模問題評価完了！結果を保存しました:")
    print("- large_scale_results.csv")
    print("- scalability_stats.csv") 
    print("- computational_complexity.csv")
    print("- large_scale_scalability.png")
    
    # 要約結果の表示
    print("\n要約結果:")
    print("=" * 50)
    max_sizes = df[df['feasible'] == True].groupby('algorithm')['problem_size'].max()
    print("各アルゴリズムの最大実行可能問題サイズ:")
    for algo, size in max_sizes.items():
        print(f"{algo:20}: {size} items")
    
    return {
        'results_df': df,
        'scalability_stats': scalability_stats,
        'complexity_df': complexity_df,
        'memory_requirements': memory_requirements
    }

if __name__ == "__main__":
    results = main()