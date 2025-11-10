"""
ナップサック問題に対する欲張り法の実装と評価
修士課程レポート課題2用実装ファイル
"""

import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class Item:
    """品物を表すデータクラス"""
    idx: int
    weight: int
    value: int
    value_density: float

def greedy_value_density(weights: List[int], values: List[int], capacity: int) -> Tuple[List[int], int, int, float]:
    """価値密度に基づく欲張り法"""
    n = len(weights)
    items = [Item(i+1, weights[i], values[i], values[i]/weights[i]) 
             for i in range(n)]
    
    # 価値密度の降順でソート
    items.sort(key=lambda x: x.value_density, reverse=True)
    
    chosen = []
    total_weight = 0
    total_value = 0
    
    start_time = time.perf_counter()
    for item in items:
        if total_weight + item.weight <= capacity:
            chosen.append(item)
            total_weight += item.weight
            total_value += item.value
    
    execution_time = time.perf_counter() - start_time
    return [item.idx for item in chosen], total_weight, total_value, execution_time

def greedy_value_only(weights: List[int], values: List[int], capacity: int) -> Tuple[List[int], int, int, float]:
    """価値のみに基づく欲張り法"""
    n = len(weights)
    items = [Item(i+1, weights[i], values[i], values[i]) for i in range(n)]
    items.sort(key=lambda x: x.value_density, reverse=True)  # 価値でソート
    
    chosen = []
    total_weight = 0
    total_value = 0
    
    start_time = time.perf_counter()
    for item in items:
        if total_weight + item.weight <= capacity:
            chosen.append(item)
            total_weight += item.weight
            total_value += item.value
    
    execution_time = time.perf_counter() - start_time
    return [item.idx for item in chosen], total_weight, total_value, execution_time

def greedy_weight_only(weights: List[int], values: List[int], capacity: int) -> Tuple[List[int], int, int, float]:
    """重量のみに基づく欲張り法（軽いものから）"""
    n = len(weights)
    items = [Item(i+1, weights[i], values[i], -weights[i]) for i in range(n)]  # 重量の昇順
    items.sort(key=lambda x: x.value_density, reverse=True)  # 負の重量でソート
    
    chosen = []
    total_weight = 0
    total_value = 0
    
    start_time = time.perf_counter()
    for item in items:
        if total_weight + item.weight <= capacity:
            chosen.append(item)
            total_weight += item.weight
            total_value += item.value
    
    execution_time = time.perf_counter() - start_time
    return [item.idx for item in chosen], total_weight, total_value, execution_time

def greedy_hybrid(weights: List[int], values: List[int], capacity: int) -> Tuple[List[int], int, int, float]:
    """ハイブリッド欲張り法（複数戦略の最良解を採用）"""
    start_time = time.perf_counter()
    
    # 各戦略の実行
    strategies = []
    
    # 価値密度戦略
    result1 = greedy_value_density(weights, values, capacity)
    strategies.append((result1[2], result1))  # (価値, 結果)
    
    # 価値のみ戦略
    result2 = greedy_value_only(weights, values, capacity)
    strategies.append((result2[2], result2))
    
    # 重量のみ戦略
    result3 = greedy_weight_only(weights, values, capacity)
    strategies.append((result3[2], result3))
    
    # 最良の解を選択（価値最大）
    best_value, best_result = max(strategies, key=lambda x: x[0])
    
    execution_time = time.perf_counter() - start_time
    return best_result[0], best_result[1], best_result[2], execution_time

def knapsack_dp(weights: List[int], values: List[int], capacity: int) -> Tuple[int, int, List[int], float]:
    """動的計画法による0-1ナップサック問題の解法"""
    n = len(weights)
    
    start_time = time.perf_counter()
    
    # DPテーブルの初期化
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # DPテーブルの構築
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]])
            else:
                dp[i][w] = dp[i-1][w]
    
    # 最適値の取得
    max_value = dp[n][capacity]
    
    # 選択された品物の復元
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i)
            w -= weights[i-1]
    
    selected_items.reverse()
    
    # 総重量の計算
    total_weight = sum(weights[i-1] for i in selected_items)
    
    execution_time = time.perf_counter() - start_time
    
    return max_value, total_weight, selected_items, execution_time

def local_search(current_solution: List[Item], all_items: List[Item], 
                 capacity: int, current_value: int, current_weight: int) -> Tuple[List[int], int, int]:
    """局所探索による解の改善"""
    best_value = current_value
    best_solution = current_solution.copy()
    
    # 1品物の追加/削除/交換を試行
    current_indices = [item.idx for item in current_solution]
    available_indices = [item.idx for item in all_items if item.idx not in current_indices]
    
    # 追加試行
    for idx in available_indices:
        item = all_items[idx-1]
        if current_weight + item.weight <= capacity:
            new_value = current_value + item.value
            if new_value > best_value:
                best_value = new_value
                best_solution = current_solution + [item]
    
    # 交換試行（1品物交換）
    for remove_idx in current_indices:
        for add_idx in available_indices:
            remove_item = all_items[remove_idx-1]
            add_item = all_items[add_idx-1]
            
            new_weight = current_weight - remove_item.weight + add_item.weight
            if new_weight <= capacity:
                new_value = current_value - remove_item.value + add_item.value
                if new_value > best_value:
                    best_value = new_value
                    best_solution = [item for item in current_solution if item.idx != remove_idx] + [add_item]
    
    return [item.idx for item in best_solution], sum(item.weight for item in best_solution), best_value

def improved_greedy(weights: List[int], values: List[int], capacity: int) -> Tuple[List[int], int, int, float]:
    """改良欲張り法：残容量考慮と部分改善"""
    n = len(weights)
    start_time = time.perf_counter()
    
    # 戦略1: 価値密度順
    density_result = greedy_value_density(weights, values, capacity)
    
    # 戦略2: 残容量を考慮した選択
    items = [Item(i+1, weights[i], values[i], values[i]/weights[i]) for i in range(n)]
    items.sort(key=lambda x: x.value_density, reverse=True)
    
    current_weight = 0
    current_value = 0
    selected = []
    
    for item in items:
        if current_weight + item.weight <= capacity:
            selected.append(item)
            current_weight += item.weight
            current_value += item.value
        else:
            # 残容量に対して価値が非常に高い単品を検討
            if item.value > current_value and item.weight <= capacity:
                selected = [item]
                current_weight = item.weight
                current_value = item.value
    
    # 戦略3: 局所探索による改善
    improved_items, improved_weight, improved_value = local_search(
        selected, items, capacity, current_value, current_weight
    )
    
    execution_time = time.perf_counter() - start_time
    
    return improved_items, improved_weight, improved_value, execution_time

def evaluate_all_methods(weights: List[int], values: List[int], capacity: int, problem_name: str):
    """すべての手法を評価して結果を表示"""
    print(f"\n{'='*50}")
    print(f"問題 {problem_name} の評価")
    print(f"{'='*50}")
    
    # 各手法の実行
    methods = [
        ("欲張り法(価値密度)", greedy_value_density),
        ("欲張り法(価値のみ)", greedy_value_only),
        ("欲張り法(重量のみ)", greedy_weight_only),
        ("ハイブリッド欲張り法", greedy_hybrid),
        ("改良欲張り法", improved_greedy),
        ("動的計画法", knapsack_dp)
    ]
    
    results = []
    
    for method_name, method_func in methods:
        if method_name == "動的計画法":
            # DPは返り値の形式が異なる
            value, weight, items, time_taken = method_func(weights, values, capacity)
        else:
            items, weight, value, time_taken = method_func(weights, values, capacity)
        
        results.append({
            'method': method_name,
            'value': value,
            'weight': weight,
            'items': items,
            'time': time_taken
        })
        
        print(f"{method_name}:")
        print(f"  総価値: {value}, 総重量: {weight}")
        print(f"  選択品目: {items}")
        print(f"  計算時間: {time_taken:.6f}秒")
        print()
    
    return results

def main():
    """メイン実行関数"""
    # 問題データの定義
    # 問題α1
    A_alpha = [3, 6, 5, 4, 8, 5, 3, 4]
    C_alpha = [7, 12, 9, 7, 13, 8, 4, 5]
    B_alpha = 25
    
    # 問題β1  
    A_beta = [3,6,5,4,8,5,3,4,3,5,6,4,8,7,11,8,14,6,12,4]
    C_beta = [7,12,9,7,13,8,4,5,3,10,7,5,6,14,5,9,6,12,5,9]
    B_beta = 55
    
    # 最悪ケースの例
    worst_weights = [10, 20, 30]
    worst_values = [100, 120, 110]
    worst_capacity = 30
    
    # 各問題の評価
    results_alpha = evaluate_all_methods(A_alpha, C_alpha, B_alpha, "α1")
    results_beta = evaluate_all_methods(A_beta, C_beta, B_beta, "β1")
    
    # 最悪ケースの評価
    print(f"\n{'='*50}")
    print("最悪ケースの評価")
    print(f"{'='*50}")
    
    greedy_worst = greedy_value_density(worst_weights, worst_values, worst_capacity)
    dp_worst = knapsack_dp(worst_weights, worst_values, worst_capacity)
    
    print(f"欲張り法: 価値={greedy_worst[2]}, 重量={greedy_worst[1]}, 品目={greedy_worst[0]}")
    print(f"動的計画法: 価値={dp_worst[0]}, 重量={dp_worst[1]}, 品目={dp_worst[2]}")
    print(f"近似比: {greedy_worst[2] / dp_worst[0]:.3f}")
    
    # 性能比較表の生成
    print(f"\n{'='*80}")
    print("性能比較まとめ")
    print(f"{'='*80}")
    print(f"{'手法':<20} {'問題':<8} {'総価値':<8} {'総重量':<8} {'計算時間(秒)':<12}")
    print(f"{'-'*80}")
    
    for result in results_alpha + results_beta:
        problem = "α1" if result in results_alpha else "β1"
        print(f"{result['method']:<20} {problem:<8} {result['value']:<8} {result['weight']:<8} {result['time']:<12.6f}")

if __name__ == "__main__":
    main()