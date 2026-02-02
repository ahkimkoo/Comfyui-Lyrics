#!/usr/bin/env python3
"""
Test to understand text preprocessing.
"""

import re

REFERENCE_TEXT = """
[Intro]

[Verse]
看着k线图心潮澎湃
梦想着别墅还有那跑车开
专家说牛市马上就到来
梭哈是一种过人的气概

[Pre-Chorus]
心跳在加速 呼吸都停摆
好像全世界为我喝彩
"""

# Step 1: Remove tags
ref_text_no_tags = re.sub(r"\[[^\]]*\]", "", REFERENCE_TEXT)

print("=== After removing tags ===")
print(repr(ref_text_no_tags))
print()

# Step 2: Replace newlines with spaces
ref_clean = ref_text_no_tags.replace("\n", " ").replace("\t", " ")

print("=== After replacing newlines ===")
print(repr(ref_clean))
print()

# Step 3: Merge multiple whitespace
ref_clean = re.sub(r"\s+", " ", ref_clean).strip()

print("=== After merging whitespace ===")
print(repr(ref_clean))
print()

# Check for comma in expected positions
text = ref_clean
pos = text.find("开")
print(f"'开' at position {pos}: '{text[max(0, pos - 2) : min(len(text), pos + 3)]}'")

pos = text.find("专家")
print(f"'专家' at position {pos}: '{text[max(0, pos - 2) : min(len(text), pos + 3)]}'")
