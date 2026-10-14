import json
import re
from collections import defaultdict

import pandas as pd

def norm(s):
    s = s.lower().replace('-', ' ')
    return re.sub(r'\s+', ' ', s)


df = pd.read_csv('data/processed/listing_sample.csv')
texts = df['remarks'].dropna().map(norm).tolist()

with open('data/processed/taxonomy.json') as f:
    terms = json.load(f)['terms']

#term id = # of remarks containing it
hits = {}
remark_has_term = [False] * len(texts)
by_category_remarks = defaultdict(set)

for t in terms:
    pat = re.compile(r'\b' + re.escape(norm(t['term'])) + r'\b')
    count = 0
    for i, text in enumerate(texts):
        if pat.search(text):
            count += 1
            remark_has_term[i] = True
            by_category_remarks[t['category']].add(i)
    hits[t['id']] = count

coverage = sum(remark_has_term) / len(texts)
print(f"Overall coverage: {coverage:.1%} of {len(texts)} remarks contain >=1 term\n")

print("Remarks mentioning each category:")
for cat, idxs in by_category_remarks.items():
    print(f"  {cat}: {len(idxs)} ({len(idxs)/len(texts):.0%})")

ranked = sorted(terms, key=lambda t: hits[t['id']], reverse=True)
print("\nTop 15 terms:")
for t in ranked[:15]:
    print(f"  {t['term']} ({t['category']}): {hits[t['id']]}")

zero = [t for t in terms if hits[t['id']] == 0]
print(f"\nZero-hit terms ({len(zero)}):")
for t in zero:
    print(f"  {t['category']}: {t['term']}")
