# Entropy & Information Gain – Play Tennis Dataset
# CCS 2424 - Adaptive Learning Task 1, Question Two (c)

import math
from collections import Counter
# ── Dataset ────────────────────────────────────────────────────────
data = [
  ('Sunny',    'Hot',  'High',   'Weak',   'No'),
  ('Sunny',    'Hot',  'High',   'Strong', 'No'),
  ('Overcast', 'Hot',  'High',   'Weak',   'Yes'),
  ('Rain',     'Mild', 'High',   'Weak',   'Yes'),
  ('Rain',     'Cool', 'Normal', 'Weak',   'Yes'),
  ('Rain',     'Cool', 'Normal', 'Strong', 'No'),
  ('Overcast', 'Cool', 'Normal', 'Strong', 'Yes'),
  ('Sunny',    'Mild', 'High',   'Weak',   'No'),
  ('Sunny',    'Cool', 'Normal', 'Weak',   'Yes'),
  ('Rain',     'Mild', 'Normal', 'Weak',   'Yes'),
  ('Sunny',    'Mild', 'Normal', 'Strong', 'Yes'),
  ('Overcast', 'Mild', 'High',   'Strong', 'Yes'),
  ('Overcast', 'Hot',  'Normal', 'Weak',   'Yes'),
  ('Rain',     'Mild', 'High',   'Strong', 'No'),
]
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
labels   = [row[4] for row in data]
def entropy(labels):
    """H(S) = -sum(p * log2(p)) for each class proportion p"""
    n = len(labels)
    counts = Counter(labels)
    return -sum((c/n) * math.log2(c/n) for c in counts.values() if c > 0)


def information_gain(data, feature_idx, labels):
    """IG(S,A) = H(S) - sum_v (|S_v|/|S|) * H(S_v)"""
    total   = len(labels)
    base_h  = entropy(labels)
    subsets = {}
    for row, label in zip(data, labels):
        val = row[feature_idx]
        subsets.setdefault(val, []).append(label)
    weighted = sum((len(v)/total) * entropy(v) for v in subsets.values())
    return base_h - weighted

# ── Calculate & display results ─────────────────────────────────────
base_entropy = entropy(labels)
print(f'Dataset entropy H(S) = {base_entropy:.4f} bits')
print('-' * 45)
gains = {}
for i, feat in enumerate(features):
    ig = information_gain(data, i, labels)
    gains[feat] = ig
    print(f'IG({feat:12s}) = {ig:.4f} bits')
best = max(gains, key=gains.get)
print('-' * 45)
print(f'Best feature to split on: {best} (IG={gains[best]:.4f})')
