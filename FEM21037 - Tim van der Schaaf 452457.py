"""
    FEM21037 - Computer Science for Business Analytics assignment    
    @author: Tim van der Schaaf
    Erasmus University Rotterdam
"""

import gc, itertools, json, re, string, time
import matplotlib.pylab as pl
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

start_time = time.time()
gc.enable()

# --------- -----FUNCTIONS ----------------------------------------------------
def load_prepare_modify_data():
    """ Opens and modifies the data to a workable format,
        it converts variations of the same term (inch, hz) to the same string,
        splits the titles and features by the regular expressions into tokens 
        and creates lists of these tokens.
    """
    f = open(r"C:\Users\timsc\Downloads\TVs-all-merged\TVs-all-merged.json")
    data = json.load(f)
    df = pd.json_normalize(data).transpose()
    # df = df.head(50)
    
    global features, model_id, shop, titles
    features, model_id, shop, titles = [], [], [], []

    for i in range(len(df)):
        for j in range(len(df[0][i])):
            features.append(df[0][i][j]['featuresMap'])
            model_id.append(df[0][i][j]['modelID'])
            titles.append(df[0][i][j]['title'])
            shop.append(df[0][i][j]['shop'])
    
    get_brands(titles)
    
    # Titles (and features) separately
    feature_values = []
    for f in features:
        for k, v in f.items():
           feature_values.append(v)   
     
    titles = replace(titles,['"',' inch','-inch','inches','Inch','-Inch',' Inch'],'inch')
    titles = replace(titles,['hertz','hz',' hz','-hz','Hertz','Hz','HZ'],'hz')
    feature_values = replace(feature_values,['"',' inch','-inch','inches','Inch','-Inch',' Inch'],'inch')
    feature_values = replace(feature_values,['hertz','hz',' hz','-hz','Hertz','Hz','HZ'],'hz')

    title_tokens = create_title_tokens(titles)
    feature_tokens = create_feature_tokens(feature_values)
     
    # Titles and features together
    features_list = []
    titles_and_features = []
    for i, f in enumerate(features):   
        full_value = ""
        for k, v in f.items():
            full_value = full_value + ' ' + str(v)
        features_list.append(full_value)
        titles_and_features.append(str(titles[i]) + ' ' + full_value)
    
    titles_and_features = replace(titles_and_features,['"',' inch','-inch','inches','Inch','-Inch',' Inch'],'inch')
    titles_and_features = replace(titles_and_features,['hertz','hz',' hz','-hz','Hertz','Hz','HZ'],'hz')
    
    title_and_feature_tokens = list(set(title_tokens + feature_tokens))
    
    return titles, title_tokens, titles_and_features, title_and_feature_tokens

def create_title_tokens(titles):
    """
    Create tokens of titles list input.
    """
    title_tokens = []
    for i in titles:
        token = re.findall("((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)", i)
        for j in token:
            title_tokens.append(j)
        title_tokens.append(i.split()[0])
    
    title_tokens = list(set(title_tokens))
    return list(set(title_tokens))

def create_feature_tokens(feature_values):
    """
    Create tokens of features values list input.
    """
    feature_tokens = []
    for i in feature_values:
        token = re.findall("((?:[a-zA-Z]+[\x21-\x7E]+[0-9]|[0-9]+[\x21-\x7E]+[a-zA-Z])[a-zA-Z0-9]*)" , i) 
        for j in token:
            feature_tokens.append(j)
    return list(set(feature_tokens))

def replace(input, replace_list, by):
    """
    Replace all values from replace_list in input list by the 'by' value.
    """
    replaced_input = [] 
    for i in input:
        for j in replace_list:
            i = re.sub(j,by,i)
        replaced_input.append(i)
    return replaced_input

def get_brands(titles):
    """
    Create a list of the brands of all titles given as input. Note: unique_brandlist
    is partly created by searching on 'Brand' key and partly by visual inspectation.
    """
    global brands
    brands = [0]*len(titles)
    unique_brandlist = ['azend','affinity','avue','coby','contex','compaq','craig','curtisyoung','dynex','elo','elite','epson','haier','hannspree','hisense','hiteker','hp','insignia','jvc','lg','lg electronics','magnavox','mitsubishi','naxa','nec','optoma','panasonic','philips','proscan','pyle','rca','samsung','sansui','sanyo','sceptre','seiki','sharp','sigmac','sony','sunbritetv','supersonic','tcl','toshiba','upstar','venturer','viewsonic','viore','vizio','westinghouse']
    
    for count, title in enumerate(titles):
        title_split = title.split()
        brands[count] = 'NA'
        for j in title_split:
            for i in unique_brandlist:
                if i == j.lower().strip().replace(",","").replace("",""):
                    brands[count] = i

def create_binary_matrix(title_tokens, titles):
    """ 
    Input title tokens and titles and return a binary matrix.
    """
    binary_vectors = []
    
    for i in range(len(titles)):
        binary_vec = np.zeros(len(title_tokens))
        titles_i_split = titles[i].split()
        for j in range(len(title_tokens)):    
            if title_tokens[j] in titles_i_split:
                binary_vec[j] = 1
        binary_vectors.append(binary_vec)
    
    binary_matrix = np.matrix(binary_vectors).transpose()

    return binary_matrix

def minhashing(binaryvec, prime, size, acoeff, bcoeff):
    """ 
    Inputs a binary vector, prime, the size of the minhash and
    two lists of randomly generated coefficients, it outputs the minhash
    of the binary vector.
    """
    minhash = [np.inf for i in range(size)]
    for index, value in enumerate(binaryvec):
        if value == 1:
            for i in range(size):
                hash = (acoeff[i]*index + bcoeff[i])%prime

                if minhash[i] > hash:
                    minhash[i] = hash

    return minhash

def concat_list(list):
    """
    Concatenates items in a list. Used for creating the key after minhashing.
    """
    result = ""
    for item in list:
        result = result+'_'+str(item[0])
    return result
    
def LSH(LSH_titles, LSH_title_tokens, maxvalue, size, prime, bands, rows, treshold, indices):

    signature_matrix = []

    # Input random array, and create binary matrix
    binary_matrix = create_binary_matrix(LSH_title_tokens, LSH_titles)    
    acoeff = np.random.randint(0, maxvalue, size)
    bcoeff = np.random.randint(0, maxvalue, size)
    
    # Create signature matrix by minhashing
    print('Create signature matrix')
    for j in range(len(LSH_titles)):
        if j %100 == 0:
            print('j ', j, ' of', len(LSH_titles))
        bin_vec = binary_matrix[:,j]
        signature_matrix.append(minhashing(bin_vec, prime, size, acoeff, bcoeff))
    signature_matrix = np.matrix(signature_matrix).transpose()  
    print('Signature matrix created')
    
    possibles = {}
    
    # For every band, create a sub signature matrix, make a key for every band
    # and put same in the same key-bucket. In terms of unique indices [0, ...., n]
    for b in range(bands):
        for v in range(len(LSH_titles)):
            sub_signature_matrix = signature_matrix[b*rows:(b+1)*rows,v]
            
            sub_signature_matrix_list = sub_signature_matrix.tolist()
            key = str(b)+concat_list(sub_signature_matrix_list)

            if key not in possibles:
                possibles[key]=[]
            possibles[key].append(v)
    
    # Create unique pairs in terms of unique indices [0, ..., n] of all that have >1
    # titles in a bucket
    possible_combis = []
    for count, key in enumerate(possibles):
        possibles[key].sort()
        
        if len(possibles[key])>1:
            combi_list = set(combinations(possibles[key],2))
            
            for element in combi_list:
                possible_combis.append(element)
    
    # Remove duplicate pairs
    possible_combis = set(possible_combis)
    
    # Now, we are going to transform them back to our known titles indices [3, 3, ...]
    possible_combis_original = {}

    for pair in possible_combis:
        pair0 = pair[0]
        pair1 = pair[1]
        
        tpair0 = indices[pair0]
        tpair1 = indices[pair1]
        
        # We make a dict with key the above obtained pair and value the pair in terms of known title indices [3,3,...]
        if tpair0 <= tpair1:
            possible_combis_original[pair] = (tpair0, tpair1)
        else:
            possible_combis_original[pair] = (tpair1, tpair0)
    
    # In terms of bootstrap indices so [0, 1, ..., ...] we calculate all possibilities
    # that are possible after brand and shop check and calculate the intersection between
    # these and the result from the LSH. This result in checked_combis, for these
    # checked combis, we will calculate the Jaccard score.
    true_possibles = calc_possibles(indices)
    global set_possible_combis_original
    
    # We create a set of the possible combis of all the keys (so the set is same as the original)
    # we create the set purely so we can calculate the intersection
    set_possible_combis_original = set(possible_combis_original)
    
    # These are all pairs in terms of [0, 1, ...] that we will calculate the Jaccard similarity for
    # these are the possible duplicates and the total of these is the comparisons we make
    pairs_to_evaluate = set_possible_combis_original.intersection(true_possibles)       

    candidates = {}
    
    total_comparisons_counter = len(pairs_to_evaluate)
    
    for count, i in enumerate(pairs_to_evaluate):
        if count%1000 == 0:
            print('count',count,' of',total_comparisons_counter)
        
        A = signature_matrix[:,i[0]]
        B = signature_matrix[:,i[1]]
        similarity = jaccard_score(A, B, average='macro')
        
        # Output is e.g. key: (0, 12) -> ((3,3), 0.9)
        if similarity >= treshold:
            candidates[i] = (possible_combis_original[i], similarity)

    return candidates, total_comparisons_counter

def get_bands_rows(t, n):
    """
    Calculate optimal bands and rows by input threshold value and number of rows.
    Output optimal values for: (threshold), rows, bands and number of rows.
    """
    
    t_list = [1]
    for r in range(1, n+1):
        b = n/r
        t_check = (1/b)**(1/r)
        t_list.append(abs(t - t_check))
        
    r_opt = t_list.index(min(t_list))
    b_opt = int(n/r_opt)
    t_opt = (1/b_opt)**(1/r_opt)
    
    n_new = b_opt*r_opt
    
    return t_opt, r_opt, b_opt, n_new
                    
def same_shop(candidate, shop_fun):
    """
    Inputs a candidate pair and its shops list, outputs if the pair is the same shop.
    """
    return shop_fun[candidate[0]] == shop_fun[candidate[1]]

def same_brand(candidate, brand_fun):
    """
    Inputs a candidate pair and its brands list, outputs if the pair is the same brand.
    """
    return brand_fun[candidate[0]] == brand_fun[candidate[1]]

def calc_possibles(indices):
    """
    This inputs the bootstrap sample indices (where repetition is possible), e.g.
    [3, 3, 1, 0] and outputs the pairs from this list that are true possibles. That is,
    have 1. same brand AND different shop OR 2. are the same (3,3). Let's say that
    (3,1) obeys 1. and (3,3) by definition obeys 2., then the output of this vector
    (in terms of the position of these pairs in the input list) is: [(0,1), (0,2), (1,2)].
    """
    
    conv_indices = [i for i in range(len(indices))] 
    all_combis = {}
    
    for pair in itertools.combinations(conv_indices, 2):
        all_combis[pair] = 0
    
    for pair in all_combis:
        # Convert to known indices
        tpair0 = indices[pair[0]]
        tpair1 = indices[pair[1]]
        
        if tpair0 <= tpair1:    
            all_combis[pair] = (tpair0, tpair1)
        else:
            all_combis[pair] = (tpair1, tpair0)
    
    true_possibles = []
    for key in all_combis.keys():
        if (same_brand(all_combis[key], brands) == True and same_shop(all_combis[key], shop) == False) or all_combis[key][0] == all_combis[key][1]:
            true_possibles.append(key)

    return set(true_possibles)

def all_dups(indices):
    """
    Calculates all duplicates based on the indices as input. If we input indices = 
    [1, ..., n] all original duplicates are obtained. If we input the bootstrap
    indices, the duplicates in the bootstrap are obtained. Output in (3,3) format.
    """
    sorted_indices = sorted(indices)
  
    all_pairs = []
    for pair in itertools.combinations(sorted_indices,2):
        all_pairs.append(pair)
    
    all_duplicates = []
    for pair in all_pairs:
        if (model_id[pair[0]] == model_id[pair[1]] and same_shop(pair, shop) == False and same_brand(pair, brands) == True) or pair[0]==pair[1]:
            all_duplicates.append(pair)
        
    return all_duplicates

def similarities(result, bootstrap_results, features, titles): 
    
    def three_gram(str_a, str_b):
        N = 3
        a_gram = {str_a[i:i+N] for i in range(len(str_a)-N+1)}
        b_gram = {str_b[i:i+N] for i in range(len(str_b)-N+1)}
    
        intersection = a_gram.intersection(b_gram)
        union = a_gram.union(b_gram)
        return float(len(intersection)) / len(union)
    
    # Key-value 3-gram similarity
    res0, res1 = result[0], result[1]
    values0, values1 = [], []
    for key in features[res0].keys():
        values0.append(features[res0][key])
    
    for key in features[res1].keys():
        values1.append(features[res1][key])
    
    full0, full1 = '', ''
    for i in values0:
        part0 = ''.join(word for word in i if word not in string.punctuation).replace(' ','').lower()
        full0 += part0

    for i in values1:
        part1 = ''.join(word for word in i if word not in string.punctuation).replace(' ','').lower()
        full1 += part1

    keyvalue_similarity = three_gram(full0, full1)
    
    # Title 3-gram similarity
    title0 = ''.join(word for word in titles[res0] if word not in string.punctuation).replace(' ','').lower()
    title1 = ''.join(word for word in titles[res1] if word not in string.punctuation).replace(' ','').lower()
    
    title_similarity = three_gram(title0, title1)

    # Jaccard similarity
    jacc_similarity = bootstrap_results[(res0, res1)][1]
    
    return [keyvalue_similarity, title_similarity, jacc_similarity]

# ---- COMPUTATIONS -----------------------------------------------------------
data = load_prepare_modify_data()

title_tokens = data[1]
titles_and_features = data[2]
title_and_feature_tokens = data[3]

# If you want to include features, set these on
# titles = titles_and_features
# title_tokens = title_and_feature_tokens
      
# Bootstrap settings
bootstrap_percentage = .63
n_bootstraps = 5
t_values = [round(t,2) for t in np.arange(0.05, 1, 0.1)]
indices = [i for i in range(len(titles))] 
train_indices, test_indices = [], []
titles_train, features_train, titles_test, features_test = {}, {}, {}, {}

# Prepare bootstrap samples
for i in range(n_bootstraps):
    train = resample(indices, replace=True, n_samples = int(len(indices) * bootstrap_percentage))
    train_indices.append(train)
    
    titles_train[i] = [titles[x] for x in train_indices[i]]
    features_train[i] = [features[x] for x in train_indices[i]]
    
    test = [x for x in indices if x not in train]
    test_indices.append(test)
    
    titles_test[i] = [titles[x] for x in test_indices[i]]
    features_test[i] = [features[x] for x in test_indices[i]]

# Create empty result dictionaries
bootstrap_full_train_results, comparisons_full_results, PQ_full_results, PC_full_results, F1_full_results, full_fraction_comparisons,  = {}, {}, {}, {}, {}, {}
bootstrap_full_test_results, full_fraction_test_comparisons, comparisons_excl_full = {}, {}, {}

# Bootstrap results
for i in range(n_bootstraps):
    bootstrap_results, comparisons_results, PQ_results, PC_results, F1_results, fraction_comparisons, fraction_test_comparisons = {}, {}, {}, {}, {}, {}, {}

    # Select correct data from bootstrap
    titles_train_iter = titles_train[i]
    title_train_tokens_iter = create_title_tokens(titles_train_iter)
    train_indices_iter = train_indices[i]
    total_comparisons_possible = len(titles_train_iter)*(len(titles_train_iter)-1)/2
    
    titles_test_iter = titles_test[i]    
    title_test_tokens_iter = create_title_tokens(titles_test_iter)
    test_indices_iter = test_indices[i]
    
    for t_iter in t_values:
        n_iter = int(len(title_train_tokens_iter)/2)
        
        get_bands_rows_iter = get_bands_rows(t_iter, n_iter)
        r_iter, b_iter, n_iter = get_bands_rows_iter[1], get_bands_rows_iter[2], get_bands_rows_iter[3]

        print('Bootstrap train #', i, ' r=', r_iter, ' b=', b_iter)
        bootstrap_results_iter = LSH(titles_train_iter, title_train_tokens_iter, maxvalue = 2000, size = n_iter, prime = 1031, bands = b_iter, rows = r_iter, treshold = t_iter, indices = train_indices[i])
        
        # These are the dicts with candidates
        bootstrap_results[t_iter] = bootstrap_results_iter[0]
        
        # This is the number of comparisons made
        comparisons_results[t_iter] = bootstrap_results_iter[1]
        
        # We compute all duplicates in original format so (3,3) is possible
        true_dupl_train = all_dups(train_indices_iter)
        
        true_dupl_found = 0
        for result in bootstrap_results[t_iter].values():
            # We obtain the (3,3) values from the bootstrap results
            pair = result[0]
            
            if pair in true_dupl_train:
                true_dupl_found +=1
    
        PQ_results[t_iter] = true_dupl_found/comparisons_results[t_iter]
        PC_results[t_iter] = true_dupl_found/len(true_dupl_train)
        
        if PQ_results[t_iter] + PC_results[t_iter] == 0:
            F1_results[t_iter] = 0
        else:
            F1_results[t_iter] = (2*PQ_results[t_iter]*PC_results[t_iter])/(PQ_results[t_iter]+PC_results[t_iter])
        
        fraction_comparisons[t_iter] = comparisons_results[t_iter]/total_comparisons_possible 
        
    bootstrap_full_train_results[i] = bootstrap_results
    comparisons_full_results[i] = comparisons_results
    PQ_full_results[i] = PQ_results
    PC_full_results[i] = PC_results
    F1_full_results[i] = F1_results
    full_fraction_comparisons[i] = fraction_comparisons

    bootstrap_results = {}
    total_comparisons_possible = len(titles_test_iter)*(len(titles_test_iter)-1)/2
    for t_iter in t_values:
        n_iter = int(len(title_test_tokens_iter)/2)
        
        get_bands_rows_iter = get_bands_rows(t_iter, n_iter)
        r_iter, b_iter, n_iter = get_bands_rows_iter[1], get_bands_rows_iter[2], get_bands_rows_iter[3]

        print('Bootstrap test #', i, ' r=', r_iter, ' b=', b_iter)
        bootstrap_results_iter = LSH(titles_test_iter, title_test_tokens_iter, maxvalue = 2000, size = n_iter, prime = 997, bands = b_iter, rows = r_iter, treshold = t_iter, indices = test_indices[i])
        bootstrap_results[t_iter] = bootstrap_results_iter[0]
        comparisons_results[t_iter] = bootstrap_results_iter[1]
        
        fraction_test_comparisons[t_iter] = comparisons_results[t_iter]/total_comparisons_possible
        
        true_dupl_test = all_dups(test_indices_iter)
        
    bootstrap_full_test_results[i] = bootstrap_results
    full_fraction_test_comparisons[i] = fraction_test_comparisons
 
# Average results 
PQ_average_results, PC_average_results, F1_average_results, full_fraction_comparisons_average = len(t_values)*[0], len(t_values)*[0], len(t_values)*[0], len(t_values)*[0]
full_fraction_test_average = len(t_values)*[0]
for index, t_iter in enumerate(t_values):
    for i in range(n_bootstraps):
        PQ_average_results[index] += 1/n_bootstraps * PQ_full_results[i][t_iter]
        PC_average_results[index] += 1/n_bootstraps * PC_full_results[i][t_iter]
        F1_average_results[index] += 1/n_bootstraps * F1_full_results[i][t_iter]
        full_fraction_comparisons_average[index] += 1/n_bootstraps * full_fraction_comparisons[i][t_iter]
        full_fraction_test_average[index] += 1/n_bootstraps * full_fraction_test_comparisons[i][t_iter]
    
# CLASSIFICATION --------------------------------------------------------------
F1_full = {}
for i in range(n_bootstraps):
    true_dupl_train = all_dups(train_indices[i])
    
    F1_result = {}
    for t_iter in t_values:
        titles_train_iter = titles_train[i]
        test_data_iter = titles_test[i]
        
        # Create train x and y matrices
        x_train, y_train, results_train = [], [], []
        for key, value in bootstrap_full_train_results[i][t_iter].items():
            all_similarities = similarities(key, bootstrap_full_train_results[i][t_iter], features_train[i], titles_train[i])
            title_similarity, kv_similarity, jacc_similarity = all_similarities[0], all_similarities[1], all_similarities[2]
            is_true_dupl = value[0] in true_dupl_train

            results_train.append(value[0])
            x_train.append([kv_similarity, title_similarity, jacc_similarity])
            y_train.append(is_true_dupl)
        
        # Create test x and y matrices
        true_dupl_test = all_dups(test_indices[i])
        x_test, y_test, results_test = [], [], []
        for key, value in bootstrap_full_test_results[i][t_iter].items():
            all_similarities = similarities(key, bootstrap_full_test_results[i][t_iter], features_test[i], titles_test[i])
            title_similarity, kv_similarity, jacc_similarity = all_similarities[0], all_similarities[1], all_similarities[2]
            is_true_dupl = value[0] in true_dupl_test
            
            results_test.append(value[0])
            x_test.append([kv_similarity, title_similarity, jacc_similarity])
            y_test.append(is_true_dupl) 
        
        # Train the logistic regression, test it on the testdata
        try:
            reg = LogisticRegression(random_state=0).fit(pd.DataFrame(x_train), np.array(y_train))
            
            reg_param = {'C': np.logspace(-3, 3, 7)}
            reg_grid = GridSearchCV(reg, reg_param, cv=3)
            reg_fit = reg_grid.fit(pd.DataFrame(x_train), np.array(y_train))
            reg_pred = reg_grid.predict(pd.DataFrame(x_test))
            
            F1 = f1_score(y_test, reg_pred, average='macro')
            F1_result[t_iter] = F1
        
        except ValueError:
            print('No 2 classes in the data')
            F1_result[t_iter] = 0
    F1_full[i] = F1_result

# Create average of classification results
F1_classification_average_results = [0]*len(t_values)
for index, t_iter in enumerate(t_values):
    for i in range(n_bootstraps):
        F1_classification_average_results[index] += 1/n_bootstraps * F1_full[i][t_iter]

# Create plots of data 
pl.figure(figsize=(20,10))
pl.plot(full_fraction_comparisons_average, PQ_average_results, '-g')
pl.xlabel('Fraction comparisons') 
pl.ylabel('Pair quality')
pl.show()

pl.figure(figsize=(20,10))
pl.plot(full_fraction_comparisons_average, PC_average_results, '-g')
pl.xlabel('Fraction comparisons') 
pl.ylabel('Pair completeness')
pl.show()

pl.figure(figsize=(20,10))
pl.plot(full_fraction_comparisons_average, F1_average_results, '-g')
pl.xlabel('Fraction comparisons') 
pl.ylabel('F1')
pl.show()

pl.figure(figsize=(20,10))
pl.plot(full_fraction_test_average, F1_classification_average_results, '-g')
pl.xlabel('Fraction comparisons') 
pl.ylabel('F1')
pl.show()
print('Code took ', time.time() - start_time, ' seconds')