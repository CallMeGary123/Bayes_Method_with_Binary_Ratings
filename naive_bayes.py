import numpy as np

R = np.array([
    [1, -1, 1, -1, 1, -1],   
    [1,  1, None, -1, -1, -1],
    [None, 1, 1, -1, -1, None], 
    [-1, -1, -1, 1, 1, 1],   
    [-1, None, -1, 1, 1, 1]  
], dtype=object)

target_user = 2   # user3
target_item = 0   # item1


def prior_probability(matrix, item, value):
    col = matrix[:, item]
    observed = [r for r in col if r is not None]
    count = sum(1 for r in observed if r == value)
    return count / len(observed)


def conditional_probability(matrix, item_j, value_j, item_i, value_i):
    num = 0
    denom = 0

    for row in matrix:
        if row[item_i] == value_i:
            if row[item_j] is not None:
                denom += 1
                if row[item_j] == value_j:
                    num += 1

    if denom == 0:
        return 0
    return num / denom


def naive_bayes_score(matrix, user, item, value):
    score = prior_probability(matrix, item, value)

    for j in range(matrix.shape[1]):
        if j == item:
            continue

        rating = matrix[user, j]
        if rating is None:
            continue

        p = conditional_probability(matrix, j, rating, item, value)
        score *= p

    return score


score_pos = naive_bayes_score(R, target_user, target_item, 1)
score_neg = naive_bayes_score(R, target_user, target_item, -1)

print(f"Score for r{target_user + 1}{target_item + 1} = 1 :", score_pos)
print(f"Score for r{target_user + 1}{target_item + 1} = -1:", score_neg)

prediction = 1 if score_pos > score_neg else -1
print(f"Predicted rating for r{target_user + 1}{target_item + 1}:", prediction)
