"""
Visualization Script - Knapsack Problem Algorithm Performance Comparison
"""
import matplotlib.pyplot as plt
import numpy as np
from greedy_knapsack import (
    greedy_value_density, greedy_value_only, greedy_weight_only,
    greedy_hybrid, improved_greedy, knapsack_dp
)
import time

# Set font to avoid character issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def plot_performance_comparison():
    """アルゴリズム性能比較の可視化"""
    # 問題データ
    A_alpha = [3, 6, 5, 4, 8, 5, 3, 4]
    C_alpha = [7, 12, 9, 7, 13, 8, 4, 5]
    B_alpha = 25
    
    A_beta = [3,6,5,4,8,5,3,4,3,5,6,4,8,7,11,8,14,6,12,4]
    C_beta = [7,12,9,7,13,8,4,5,3,10,7,5,6,14,5,9,6,12,5,9]
    B_beta = 55

    # アルゴリズムリスト
    methods = [
        ("Greedy(Value Density)", greedy_value_density),
        ("Greedy(Value Only)", greedy_value_only),
        ("Greedy(Weight Only)", greedy_weight_only),
        ("Hybrid Greedy", greedy_hybrid),
        ("Improved Greedy", improved_greedy),
        ("Dynamic Programming", knapsack_dp)
    ]

    # 結果格納用
    alpha_values = []
    alpha_times = []
    beta_values = []
    beta_times = []
    method_names = []

    for name, method in methods:
        method_names.append(name)
        
        # α1問題
        if name == "Dynamic Programming":
            value, _, _, time_taken = method(A_alpha, C_alpha, B_alpha)
        else:
            _, _, value, time_taken = method(A_alpha, C_alpha, B_alpha)
        alpha_values.append(value)
        alpha_times.append(time_taken * 1000)  # ミリ秒に変換
        
        # β1問題
        if name == "Dynamic Programming":
            value, _, _, time_taken = method(A_beta, C_beta, B_beta)
        else:
            _, _, value, time_taken = method(A_beta, C_beta, B_beta)
        beta_values.append(value)
        beta_times.append(time_taken * 1000)  # ミリ秒に変換

    # プロット設定
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 総価値比較（α1）
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, alpha_values, width, label='α1', alpha=0.8)
    bars2 = ax1.bar(x + width/2, beta_values, width, label='β1', alpha=0.8)
    
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Total Value')
    ax1.set_title('Total Value Comparison by Algorithm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=45)
    ax1.legend()
    
    # バー上に数値を表示
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')

    # 2. 計算時間比較（α1）
    ax2.bar(method_names, alpha_times, alpha=0.7, color='orange')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Execution Time (milliseconds)')
    ax2.set_title('Execution Time Comparison for Problem α1')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names, rotation=45)
    
    # 3. 計算時間比較（β1）
    ax3.bar(method_names, beta_times, alpha=0.7, color='green')
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Execution Time (milliseconds)')
    ax3.set_title('Execution Time Comparison for Problem β1')
    ax3.set_xticks(x)
    ax3.set_xticklabels(method_names, rotation=45)

    # 4. 近似比の比較
    optimal_alpha = max(alpha_values)
    optimal_beta = max(beta_values)
    
    alpha_ratios = [v/optimal_alpha for v in alpha_values]
    beta_ratios = [v/optimal_beta for v in beta_values]
    
    bars3 = ax4.bar(x - width/2, alpha_ratios, width, label='α1', alpha=0.8)
    bars4 = ax4.bar(x + width/2, beta_ratios, width, label='β1', alpha=0.8)
    
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Approximation Ratio (Optimal=1.0)')
    ax4.set_title('Approximation Ratio by Algorithm')
    ax4.set_xticks(x)
    ax4.set_xticklabels(method_names, rotation=45)
    ax4.legend()
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Optimal')

    plt.tight_layout()
    plt.savefig('algorithm_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_scalability():
    """Scalability visualization"""
    # 問題サイズを変えて実行時間を計測
    sizes = [10, 20, 30, 40, 50]
    capacity_base = 50
    
    greedy_times = []
    dp_times = []
    
    for size in sizes:
        # ランダムな問題インスタンス生成
        np.random.seed(42)
        weights = np.random.randint(1, 20, size)
        values = np.random.randint(1, 50, size)
        capacity = capacity_base + size * 2
        
        # 欲張り法の実行時間
        start_time = time.perf_counter()
        greedy_value_density(weights.tolist(), values.tolist(), capacity)
        greedy_times.append((time.perf_counter() - start_time) * 1000)
        
        # 動的計画法の実行時間（小さい問題のみ）
        if size <= 30:
            start_time = time.perf_counter()
            knapsack_dp(weights.tolist(), values.tolist(), capacity)
            dp_times.append((time.perf_counter() - start_time) * 1000)
        else:
            dp_times.append(None)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes[:len(dp_times)], dp_times, 'o-', label='Dynamic Programming', linewidth=2)
    plt.plot(sizes, greedy_times, 's-', label='Greedy(Value Density)', linewidth=2)
    
    plt.xlabel('Problem Size (Number of Items)')
    plt.ylabel('Execution Time (milliseconds)')
    plt.title('Algorithm Scalability Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('algorithm_scalability.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_algorithm_flowchart():
    """Algorithm flowchart description text"""
    flowchart_text = """
Greedy Algorithm (Value Density) Flowchart:

Start
↓
Sort items by value density (value/weight) in descending order
↓
Initialize empty knapsack (total weight=0, total value=0)
↓
Process sorted item list:
   ↓
   Can current item be added without exceeding capacity?
   ├─ Yes → Add item, update total weight and value
   └─ No → Move to next item
↓
All items processed
↓
Output selected items and total value
↓
End

Dynamic Programming Flowchart:

Start
↓
Initialize DP table (size: (n+1) × (capacity+1))
↓
For item i=1 to n:
   For weight w=1 to capacity:
   ↓
   Item i weight <= w?
   ├─ Yes → DP[i][w] = max(DP[i-1][w], value[i] + DP[i-1][w-weight[i]])
   └─ No → DP[i][w] = DP[i-1][w]
↓
Optimal value = DP[n][capacity]
↓
Backtrack to recover selected items
↓
Output results
↓
End
"""
    with open('algorithm_flowcharts.txt', 'w', encoding='utf-8') as f:
        f.write(flowchart_text)
    print("Flowcharts saved to 'algorithm_flowcharts.txt'")

if __name__ == "__main__":
    print("Running visualization script...")
    
    # Algorithm performance comparison plot
    plot_performance_comparison()
    print("Performance comparison graph saved to 'algorithm_performance_comparison.png'")
    
    # Scalability plot
    plot_scalability()
    print("Scalability graph saved to 'algorithm_scalability.png'")
    
    # Flowchart creation
    create_algorithm_flowchart()
    
    print("All visualizations completed")