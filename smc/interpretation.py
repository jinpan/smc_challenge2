import pandas as pd


def print_precision_recall_stats(df):
  for space_group in df['actual'].unique():
    num_actual = len(df[df['actual']==space_group])
    num_predicted = len(df[df['predicted']==space_group])
    num_correct = len(df[(df['actual']==space_group)&(df['actual']==df['predicted'])])

    if num_predicted == 0:
        precision = 1.
    else:
        precision = num_correct / num_predicted
    recall = num_correct / num_actual

    print(f"{space_group:3d}: precision: {100*precision:0.1f}% recall: {100*recall:0.1f}")
