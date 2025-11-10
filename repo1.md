[[853fc0ef87e33e0e.pdf]]
知識::[[知能322 数理計画とアルゴリズム レポート1 提出用]]>>[[知能322_数理計画とアルゴリズム_レポート1_提出用.pdf]]

## （1）ナップサック問題および輸送問題と同様に定式化して解くことができる新しい問題を、日常生活の中から例を取って１つずつ作問せよ。また、その問題を詳しく説明するとともに定式化した結果のみを示せ。定式化した問題を解く必要はない。


### ナップサック

スーパーでの買い物。商品集合$(i)$ があり、各商品$(i)$ には価格 (p_i) と体積（カゴ占有） (v_i)、主観的満足度 (u_i) がある。手元の予算は (B)、買い物カゴの容量は (V)。どの商品を買うか二値で選び、予算と容量の範囲で満足度合計を最大化したい。解けとは言っていないので、きっちり定式化だけ示す。

定式化
集合・パラメータ：

- $(I)$：商品集合
    
- $(p_i>0)$：商品$(i)$ の価格
    
- $(v_i>0)$：商品$(i)$ の体積
    
- $(u_i\ge0)$：商品$(i)$ の満足度
    
- $(B>0)$：予算
    
- $(V>0)$：カゴ容量
    

変数：

- $(x_i\in{0,1})$（商品$(i)$ を買うなら1、買わないなら0）
    

目的関数：  
$$[  
\max \sum_{i\in I} u_i x_i  
]$$

制約：  
$$[  
\sum_{i\in I} p_i x_i \le B,\qquad  
\sum_{i\in I} v_i x_i \le V,\qquad  
x_i\in{0,1}\ \forall i\in I.  
]$$

---

# 輸送型

近所で単一品目（例：米5kg袋）を共同購入する。複数の店舗集合 $(S)$があり、各店舗 $(s)$の在庫は $(a_s)$ 袋。参加世帯の集合 $(H)$ があり、各世帯 $(h)$ は需要 $(b_h)$ 袋を希望する。店舗 $(s)$ から世帯 $(h)$ に1袋振り分けると、価格や受取手間をまとめた単位コスト $(c_{sh})$ が発生する。総コストを最小化するように、どの店舗からどの世帯へ何袋割り当てるかを決める。ありがちな輸送問題の顔をしているが、現実の面倒さはコストに押し込める。

定式化（結果のみ）  
集合・パラメータ：

- $(S)$：店舗（供給元）集合
    
- $(H)$：世帯（需要先）集合
    
- $(a_s\ge0)$：店舗 $(s)$ の在庫（供給量）
    
- $(b_h\ge0)$：世帯 $(h)$ の必要量
    
- $(c_{sh}\ge0)$：店舗 $(s)$ から世帯 $(h)$ へ1袋配分するコスト
    

変数：

- $(x_{sh}\ge0)$：店舗 $(s)$ から世帯 $(h)$ へ配分する数量（袋）
    

目的関数：  
$$[  
\min \sum_{s\in S}\sum_{h\in H} c_{sh},x_{sh}  
]$$

制約：  
$$[  
\sum_{h\in H} x_{sh} \le a_s \quad (\forall s\in S),\qquad  
\sum_{s\in S} x_{sh} = b_h \quad (\forall h\in H),\qquad  
x_{sh}\ge 0.  
]$$


## （2）連立一次方程式を解く方法を調べ、プログラムとして実装せよ。 プログラム言語は何でも良い。適当なｎ元連立一次方程式を解き、 作成したプログラムが正しいことを示せ。ただし、ｎ≦４とする。

``` python
import numpy as np

def solve_and_verify_with_numpy(A, b, x_true=None):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x_hat = np.linalg.solve(A, b)
    resid = A @ x_hat - b
    resid_inf = np.max(np.abs(resid))
    if x_true is not None:
        x_true = np.array(x_true, dtype=float)
        err_inf = np.max(np.abs(x_hat - x_true))
    else:
        err_inf = None
    return x_hat, resid_inf, err_inf

# 例1: 4元（既知解で検証）
A4 = np.array([
    [3.0,  2.0, -1.0, 4.0],
    [2.0, -2.0,  4.0, 1.0],
    [-1.0, 0.5, -1.0, 1.0],
    [5.0,  2.0,  0.0, 3.0],
], dtype=float)
x_true4 = np.array([2.0, -1.0, 3.0, 0.5], dtype=float)
b4 = A4 @ x_true4
x_hat4, resid4, err4 = solve_and_verify_with_numpy(A4, b4, x_true4)

# 例2: 3元（ランダム生成、既知解で検証）
rng = np.random.default_rng(42)
A3 = rng.normal(size=(3,3))
A3 += np.eye(3) * 2.0
x_true3 = np.array([1.0, -2.0, 0.5])
b3 = A3 @ x_true3
x_hat3, resid3, err3 = solve_and_verify_with_numpy(A3, b3, x_true3)

```


```
【例1: 4元】
解 x_hat: [1.9999999999999951, -0.9999999999999845, 3.0000000000000107, 0.49999999999999845]
真の解 x_true: [2.0, -1.0, 3.0, 0.5]
∞ノルム残差: 1.7763568394002505e-15
∞ノルム誤差: 1.554312234475219e-14

【例2: 3元】
解 x_hat: [1.0000000000000004, -1.9999999999999998, 0.5000000000000001]
∞ノルム残差: 3.0531133177191805e-16
∞ノルム誤差: 4.440892098500626e-16
```

## （3）下記α１のナップサック問題を解くための総当り法のプログラムを作れ。 プログラミング言語は何でも良いがソースリストを示せ。 


![[8f8a106b269edc4c544dcd0ffbb6fdc9_MD5.png]]

```python
# α1 ナップサック（総当り）
# 品目 i = 1..8
# 重量 Ai = [3,6,5,4,8,5,3,4]
# 価格 Ci = [7,12,9,7,13,8,4,5]
# 容量 B = 25

from typing import List, Tuple

def brute_force_knapsack(weights: List[int], values: List[int], capacity: int
                         ) -> Tuple[List[int], int, int]:
    """
    総当りで 0/1 ナップサックを解く。
    戻り値: (選択した品目番号のリスト(1始まり), 総重量, 総価値)
    """
    n = len(weights)
    best_value = -1
    best_weight = 0
    best_mask = 0

    # すべての部分集合をビットで総当り
    for mask in range(1 << n):  # 0..2^n-1
        w = 0
        v = 0
        for i in range(n):
            if (mask >> i) & 1:
                w += weights[i]
                v += values[i]
        # 容量以下かつ価値最大（同価値なら軽い方を優先）
        if w <= capacity and (v > best_value or (v == best_value and w < best_weight)):
            best_value = v
            best_weight = w
            best_mask = mask

    chosen = [i + 1 for i in range(n) if (best_mask >> i) & 1]
    return chosen, best_weight, best_value


if __name__ == "__main__":
    A = [3, 6, 5, 4, 8, 5, 3, 4]   # 重量 Ai
    C = [7,12, 9, 7,13, 8, 4, 5]   # 価格(価値) Ci
    B = 25                         # 容量 B

    items, total_w, total_v = brute_force_knapsack(A, C, B)

    print("選択品目(番号):", items)
    print("総重量:", total_w)
    print("総価値:", total_v)

```

```
選択品目(番号): [1, 2, 3, 5, 7]
総重量: 25
総価値: 45
```

## （4）下記β１のナップサック問題（品物20個の場合）を総当たり法で解け。 また、α１とβ１の 総当たり数と計算時間をそれぞれ比較検討せよ。


![[fd30a1fc1efa8d8d8228d27d37b5b261_MD5.png]]

```python
from typing import List, Tuple
from time import perf_counter

def brute_force_knapsack(weights: List[int], values: List[int], capacity: int
                         ) -> Tuple[List[int], int, int, int]:
    """
    総当りで 0/1 ナップサックを解く。
    戻り値: (選択インデックス(1始まり), 総重量, 総価値, 評価した組合せ数=2^n)
    """
    n = len(weights)
    N = 1 << n  # 総当たり数
    best_value = -1
    best_weight = 0
    best_mask = 0

    for mask in range(N):
        w = 0
        v = 0
        for i in range(n):
            if (mask >> i) & 1:
                w += weights[i]
                v += values[i]
        if w <= capacity and (v > best_value or (v == best_value and w < best_weight)):
            best_value = v
            best_weight = w
            best_mask = mask

    chosen = [i + 1 for i in range(n) if (best_mask >> i) & 1]
    return chosen, best_weight, best_value, N

def run_case(name: str, weights, values, capacity):
    t0 = perf_counter()
    chosen, tw, tv, N = brute_force_knapsack(weights, values, capacity)
    dt = perf_counter() - t0
    return name, chosen, tw, tv, N, dt

# α1
A_alpha = [3, 6, 5, 4, 8, 5, 3, 4]
C_alpha = [7, 12, 9, 7, 13, 8, 4, 5]
B_alpha = 25

# β1
A_beta = [3,6,5,4,8,5,3,4,3,5,6,4,8,7,11,8,14,6,12,4]
C_beta = [7,12,9,7,13,8,4,5,3,10,7,5,6,14,5,9,6,12,5,9]
B_beta = 55

print(run_case("α1", A_alpha, C_alpha, B_alpha))
print(run_case("β1", A_beta, C_beta, B_beta))

```

# 結果（要点）

- α1（n=8, B=25）  
    最良選択: [1, 2, 3, 5, 7]  
    総重量: 25 / 総価値: 45  
    総当たり数: 256 = 2^8  
    実測計算時間: 約 0.000287 秒
    
- β1（n=20, B=55）  
    最良選択: [1, 2, 3, 4, 5, 7, 8, 10, 14, 18, 20]  
    総重量: 55 / 総価値: 102  
    総当たり数: 1,048,576 = 2^20  
    実測計算時間: 約 3.40 秒
    
- 比較  
    組合せ数の増加は 4096 倍（2^(20−8)）。計算時間は実測で 約 1.19×10^4 倍。理論 O(2nn)O(2^n n)O(2nn) のとおり、n がちょっと増えるだけで時間が溶ける。20 でこのざま。30 なら 2^10 ≈ 1024 倍さらに重くなる。総当りは正しいが無慈悲だ。
    

# 使ったデータ

- α1: 重量 A=[3,6,5,4,8,5,3,4]A=[3,6,5,4,8,5,3,4]A=[3,6,5,4,8,5,3,4], 価値 C=[7,12,9,7,13,8,4,5]C=[7,12,9,7,13,8,4,5]C=[7,12,9,7,13,8,4,5], 容量 B=25B=25B=25
    
- β1: 重量  
    A=[3,6,5,4,8,5,3,4,3,5,6,4,8,7,11,8,14,6,12,4]A=[3,6,5,4,8,5,3,4,3,5,6,4,8,7,11,8,14,6,12,4]A=[3,6,5,4,8,5,3,4,3,5,6,4,8,7,11,8,14,6,12,4],  
    価値  
    C=[7,12,9,7,13,8,4,5,3,10,7,5,6,14,5,9,6,12,5,9]C=[7,12,9,7,13,8,4,5,3,10,7,5,6,14,5,9,6,12,5,9]C=[7,12,9,7,13,8,4,5,3,10,7,5,6,14,5,9,6,12,5,9], 容量 B=55B=55B=55


## （5）β１の場合の総当たり計算を高速化する工夫を考えて実行せよ。 その工夫によってどのくらい高速化されたか示せ。

```python
# β1 の 0/1 ナップサックを「総当り」前提で高速化する工夫を2つ実装し、
# ベースライン（素朴な全列挙）と計算速度を比較する。
#
# 工夫1: Gray code による全列挙（各ステップで1ビットしか変化しないので、
#        重量・価値の更新が O(1)。総当りはそのまま 2^n 通り）
# 工夫2: 半分全列挙（Meet-in-the-Middle）。2^(n/2) + 2^(n/2) 通りの列挙で
#        正確解を求める。全探索系の高速化として定番。
#
# どちらも最適解を返す。速度だけを比べるため、枝刈りなどはしない。

from time import perf_counter
from bisect import bisect_right
from typing import List, Tuple, Dict
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

# データ（β1）
W = [3,6,5,4,8,5,3,4,3,5,6,4,8,7,11,8,14,6,12,4]
V = [7,12,9,7,13,8,4,5,3,10,7,5,6,14,5,9,6,12,5,9]
B = 55

def brute_force_baseline(weights: List[int], values: List[int], capacity: int):
    n = len(weights)
    N = 1 << n
    best_v = -1
    best_w = 0
    best_mask = 0
    t0 = perf_counter()
    for mask in range(N):
        w = 0
        v = 0
        for i in range(n):
            if (mask >> i) & 1:
                w += weights[i]
                v += values[i]
        if w <= capacity and (v > best_v or (v == best_v and w < best_w)):
            best_v = v
            best_w = w
            best_mask = mask
    dt = perf_counter() - t0
    items = [i+1 for i in range(n) if (best_mask >> i) & 1]
    return {"method":"baseline_bitloop","n":n,"N":N,"time":dt,"value":best_v,"weight":best_w,"items":items}

def brute_force_graycode(weights: List[int], values: List[int], capacity: int):
    n = len(weights)
    N = 1 << n
    best_v = -1
    best_w = 0
    best_mask = 0
    w = 0
    v = 0
    prev = 0
    t0 = perf_counter()
    for k in range(N):
        g = k ^ (k >> 1)  # Gray code
        if k == 0:
            pass  # g=0 のときは w=v=0 のまま
        else:
            d = g ^ prev          # 変化した唯一のビット
            i = d.bit_length()-1  # そのインデックス（0始まり）
            if (g >> i) & 1:      # 立ったなら追加、消えたなら減算
                w += weights[i]; v += values[i]
            else:
                w -= weights[i]; v -= values[i]
        if w <= capacity and (v > best_v or (v == best_v and w < best_w)):
            best_v = v
            best_w = w
            best_mask = g
        prev = g
    dt = perf_counter() - t0
    items = [i+1 for i in range(n) if (best_mask >> i) & 1]
    return {"method":"graycode_fullenum","n":n,"N":N,"time":dt,"value":best_v,"weight":best_w,"items":items}

def mitm_knapsack(weights: List[int], values: List[int], capacity: int):
    n = len(weights)
    mid = n // 2
    W1, V1 = weights[:mid], values[:mid]
    W2, V2 = weights[mid:], values[mid:]
    N1, N2 = 1 << len(W1), 1 << len(W2)

    # 前半の全列挙
    L1 = []
    for m in range(N1):
        w = 0; v = 0
        for i in range(len(W1)):
            if (m >> i) & 1:
                w += W1[i]; v += V1[i]
        if w <= capacity:
            L1.append((w, v, m))
    # 重複重さで価値最大のみに圧縮し、さらに劣後解を削除（重さ増で価値非減）
    L1.sort()  # weight, then value
    L1_comp = []
    max_v = -1
    for w, v, m in L1:
        if v > max_v:
            L1_comp.append((w, v, m))
            max_v = v
    W1_sorted = [w for w, v, m in L1_comp]
    V1_prefixmax = [v for w, v, m in L1_comp]
    M1_kept = [m for w, v, m in L1_comp]

    # 後半の全列挙 + 二分探索で最良対を探す
    best_v = -1
    best_w = 0
    best_pair = (0, 0)  # (m1, m2)
    for m2 in range(N2):
        w2 = 0; v2 = 0
        for i in range(len(W2)):
            if (m2 >> i) & 1:
                w2 += W2[i]; v2 += V2[i]
        if w2 > capacity:
            continue
        rest = capacity - w2
        idx = bisect_right(W1_sorted, rest) - 1
        if idx >= 0:
            v = v2 + V1_prefixmax[idx]
            w = w2 + W1_sorted[idx]
            if v > best_v or (v == best_v and w < best_w):
                best_v = v
                best_w = w
                best_pair = (M1_kept[idx], m2)

    # 復元
    m1, m2 = best_pair
    items = []
    for i in range(len(W1)):
        if (m1 >> i) & 1:
            items.append(i+1)
    for i in range(len(W2)):
        if (m2 >> i) & 1:
            items.append(mid + i + 1)

    dt = None  # 時間は外側で計測
    return {"method":"mitm_half_enum","n":n,"N":N1+N2,"time":0.0,"value":best_v,"weight":best_w,"items":sorted(items)}

# 実測
rows = []

t0 = perf_counter()
res_base = brute_force_baseline(W, V, B)
t1 = perf_counter()
res_gray = brute_force_graycode(W, V, B)
t2 = perf_counter()

# MITM の時間は全体を測定
t3 = perf_counter()
res_mitm = mitm_knapsack(W, V, B)
t4 = perf_counter()

res_mitm["time"] = (t4 - t3)

rows.extend([res_base, res_gray, res_mitm])

df = pd.DataFrame(rows)
df["speedup_vs_baseline"] = df["time"].iloc[0] / df["time"]
display_dataframe_to_user("β1 総当たりの高速化比較", df)

print(df.to_string(index=False))

```



1. 素朴な総当り（ベースライン）  
    ビットを毎回なめて重さ・価値を合計。脳筋。計算量は $(O(2^n n))$。
    
2. Gray code 全列挙  
    隣接する組合せが「1ビットだけ違う」順に列挙。  
    各手番で増減する品目が1個なので、重さ・価値の更新が $O(1)$。  
    同じ 2^n 通りを評価する「総当り」のまま、定数倍を大幅短縮。
    
3. 半分全列挙（Meet-in-the-Middle）  
    前半10品と後半10品に分けて、それぞれの部分集合を列挙。  
    前半は重さ昇順で非劣解だけ残し、後半の各集合に対して残り容量に入る前半の最良を二分探索で合体。  
    計算量はざっくり $(O(2^{n/2}\log 2^{n/2}))$。全探索系の高速化として定番。
    



# β1 の実測結果

同一マシン・同一プロセスで測定。価値の最適値はすべて 102、重量は 55。

|方法|評価した組合せ数|時間[秒]|ベースライン比スピードアップ|解（品目番号）|
|---|--:|--:|--:|---|
|素朴な総当り|1,048,576|3.595|1.00×|[1,2,3,4,5,7,8,10,14,18,20]|
|Gray code 全列挙|1,048,576|0.441|8.15×|[1,2,3,4,5,7,10,12,14,18,20]|
|半分全列挙 (MITM)|2,048（=2×2^10）|0.00429|838×|[1,2,3,4,5,7,8,10,14,18,20]|

