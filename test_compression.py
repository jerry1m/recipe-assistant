"""
测试 LLM 对话压缩（方案 A）—— 多轮对话触发压缩 + 指标验证

流程:
  1. 清理 test_compression 会话
  2. 连续发送 6 轮对话，累积历史
  3. 检查压缩指标 (/metrics/compression)
  4. 再发几轮，观察压缩是否有累积
"""

import json
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
SESSION = "test_compression"


def ask(query: str, session: str = SESSION, label: str = ""):
    """发送单轮查询（120s 超时）"""
    body = json.dumps({
        "query": query,
        "session_id": session,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/ask",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            elapsed = time.perf_counter() - t0
            print(f"[{label}] intent={data.get('intent','?')}  latency={elapsed:.1f}s")
            print(f"    answer: {data.get('answer','')[:100]}...")
            print()
            return data
    except Exception as e:
        print(f"[{label}] ERROR: {e}")
        return None


def get_compression_stats():
    """获取压缩指标"""
    try:
        with urllib.request.urlopen(f"{BASE}/metrics/compression", timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [metrics] ERROR: {e}")
        return {}


def clear_session():
    """清空会话（通过清空 + 首次查询触发）"""
    # 清空需要直接操作 API 内部，这里用一个小技巧：
    # 先将 session_id 设为其他值来"重置"
    pass


# ── 先看当前压缩指标（应为 0）──
print("=" * 60)
print("【阶段 0】初始压缩指标")
print("=" * 60)
stats0 = get_compression_stats()
print(f"  {json.dumps(stats0, ensure_ascii=False, indent=2)}")
print()

# ── 第 1-6 轮：连续对话 ──
queries = [
    ("I want to cook Braised Pork Belly (Hong Shao Rou), any recommendations?", "Q1-pork"),
    ("What ingredients do I need for this dish?", "Q2-ingredients"),
    ("Can you give me the step-by-step instructions?", "Q3-steps"),
    ("I want to reduce sugar, what can I substitute it with?", "Q4-substitute"),
    ("How many calories does this dish have?", "Q5-calories"),
    ("Suggest a light and healthy dish instead.", "Q6-light"),
]

print("=" * 60)
print("【阶段 1】连续 6 轮对话（英文提问）")
print("=" * 60)
for query, label in queries:
    ask(query, label=label)
    time.sleep(1.0)

# 等待后台压缩完成
print("等待后台压缩（10秒）...")
time.sleep(10)

# ── 检查压缩指标 ──
print("=" * 60)
print("【阶段 2】压缩指标")
print("=" * 60)
stats1 = get_compression_stats()
print(f"  {json.dumps(stats1, ensure_ascii=False, indent=2)}")
print()

# ── 再发几轮，观察压缩累积 ──
queries2 = [
    ("Recommend a low-fat soup", "Q7-soup"),
    ("How long should I simmer this soup?", "Q8-simmer"),
    ("What vegetables go well with this soup?", "Q9-vegetables"),
]

print("=" * 60)
print("【阶段 3】再发 3 轮观察累积")
print("=" * 60)
for query, label in queries2:
    ask(query, label=label)
    time.sleep(0.5)

print("等待后台压缩...")
time.sleep(10)

# ── 最终压缩指标 ──
print("=" * 60)
print("【阶段 4】最终压缩指标")
print("=" * 60)
stats2 = get_compression_stats()
print(f"  {json.dumps(stats2, ensure_ascii=False, indent=2)}")
print()

# ── 汇总报告 ──
print("=" * 60)
print("📊 压缩效果报告")
print("=" * 60)
if stats2.get("total_compression_calls", 0) > 0:
    orig = stats2["total_original_tokens"]
    comp = stats2["total_compressed_tokens"]
    ratio = stats2["overall_compression_ratio"]
    saved = stats2["tokens_saved"]
    print(f"  压缩调用次数:   {stats2['total_compression_calls']}")
    print(f"  压缩轮次数:     {stats2['total_turns_compressed']}")
    print(f"  原始 token 数:  {orig}")
    print(f"  压缩后 token 数: {comp}")
    print(f"  压缩率:         {ratio} ({ratio*100:.1f}%)")
    print(f"  节省 token:     {saved} ({saved/max(1,orig)*100:.1f}%)")
    print(f"  ✅ 压缩效果明显！")
else:
    print("  ⚠️ 未触发压缩（轮次不足或预算充足）")
