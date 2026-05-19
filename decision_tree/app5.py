from math import log2

# =====================================================
# 1. Original Dataset
# =====================================================

# Total samples:
# A = 6
# B = 4

parent = [6, 4]

# After split:
# Left  node -> A=4, B=1
# Right node -> A=2, B=3

# left = [4, 1]
# right = [2, 3]

left = [5,0]
right = [1,4]

# =====================================================
# 2. Gini Impurity
# =====================================================

def gini(counts):

    total = sum(counts)

    gini_value = 1 - sum(
        (c / total) ** 2
        for c in counts
    )

    return gini_value

# =====================================================
# 3. Entropy
# =====================================================

def entropy(counts):

    total = sum(counts)

    entropy_value = 0

    for c in counts:

        if c == 0:
            continue

        p = c / total

        entropy_value -= p * log2(p)

    return entropy_value

# =====================================================
# 4. Weighted Impurity
# =====================================================

def weighted_average(children, metric_function):

    total_samples = sum(sum(child) for child in children)

    weighted_result = 0

    for child in children:

        child_total = sum(child)

        weighted_result += (
            child_total / total_samples
        ) * metric_function(child)

    return weighted_result

# =====================================================
# 5. Calculate Parent Metrics
# =====================================================

print("\n================ Parent Node ================\n")

parent_gini = gini(parent)
parent_entropy = entropy(parent)

print("Parent Gini:", parent_gini)
print("Parent Entropy:", parent_entropy)

# =====================================================
# 6. Calculate Child Metrics
# =====================================================

print("\n================ Child Nodes ================\n")

left_gini = gini(left)
right_gini = gini(right)

left_entropy = entropy(left)
right_entropy = entropy(right)

print("Left Gini:", left_gini)
print("Right Gini:", right_gini)

print()

print("Left Entropy:", left_entropy)
print("Right Entropy:", right_entropy)

# =====================================================
# 7. Weighted After Split
# =====================================================

print("\n================ After Split ================\n")

gini_after = weighted_average(
    [left, right],
    gini
)

entropy_after = weighted_average(
    [left, right],
    entropy
)

print("Weighted Gini After Split:", gini_after)
print("Weighted Entropy After Split:", entropy_after)

# =====================================================
# 8. Information Gain
# =====================================================

print("\n================ Information Gain ================\n")

information_gain = (
    parent_entropy - entropy_after
)

print("Information Gain:", information_gain)

# =====================================================
# 9. Split Information
# =====================================================

print("\n================ Split Information ================\n")

total = 10

left_ratio = 5 / total
right_ratio = 5 / total

split_information = -(
    left_ratio * log2(left_ratio)
    +
    right_ratio * log2(right_ratio)
)

print("Split Information:", split_information)

# =====================================================
# 10. Gain Ratio
# =====================================================

print("\n================ Gain Ratio ================\n")

gain_ratio = (
    information_gain / split_information
)

print("Gain Ratio:", gain_ratio)