import torch
import numpy as np
import pandas as pd


def fold_weighted_mean(scores, weights):
    return (scores * weights).sum()