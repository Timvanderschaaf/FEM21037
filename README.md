# FEM21037
Computer Science for Business Analytics individual paper

The code presented in this repository is meant for duplicate detection for products in aggregated (different) webshopdata. The code itself contains comments about the steps taken. Data is attached in the repository. The lines of code include:

- Line 7 - 19: imports of packages
- Line 20 - 394: various functions that are used
- Line 396: data initalization and modification
- Line 397 - 431: bootstrap settings and bootstrap data initalization
- Line 432 - 519: LSH (for flow, please read below)
- Line 520 - 575: classification (for flow, please read below)  
- Line 576 - 600: plots

Line 432 - 519: LSH flow
1. For every bootstrap, obtain correct train and test sample indices from pre-defined bootstrap sample.
2. Then within every bootstrap, for threshold values 0.05, 0.15, ..., 0.95, get the optimal value for the bands, rows and number of minhashes using the threshold value.
3. With these values, calculate the bootstrap results for the train sample by means of the LSH function.
  3.1. Create binary vectors from the tokens given as input.
  3.2. Create the signature matrix based on these binary vectors.
  3.3. For every band, for every product, find out what products hash to the same bucket. If multiple products hash to the same bucket, all unique combinations within     these products are considered as possible pairs.
  3.4. For every possible pair, evaluate whether the two products have the same brand and are from a different webshop, or are identical by comparing it to all pairs       that obey this property.
  3.5. Consider these products as the products that need to be compared. Calculate the Jaccard similarity based on the signature matrix.
  3.6. The pairs that pass the threshold value based on their Jaccard similarity, are considered candidate pairs.
4. From these candidate pairs, check whether they are true duplicates by comparing them based on their modelID.
5. Calculate pair completeness, pair quality and F1 score. Store all in a dictionary.
6. Perform step 3 for the test sample.
7. Calculate for both train and test sample the average pair completeness, pair quality and F1 score. Store in a list.

Line 520 - 575: classification flow
1. For every train and test bootstrap, for every threshold value, calculate the title similarity, feature value similarity and obtain the Jaccard similarity from LSH.
2. Train a logistic regression algorithm and perform a grid search on parameter c. Calculate F1 score.
3. Calculate for both train and test sample the average F1 score.
