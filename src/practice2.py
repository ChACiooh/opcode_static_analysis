import numpy as np
import os, glob	# to set directory
import os.path
from sklearn.model_selection import train_test_split
from sklearn import svm

# codon mapping
instr_code = dict()
f = open('instrlist.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
	c = format(i+1, 'X')	# hex
	instr_code[lines[i]] = c
f.close()

def directory_mapping(d, files):
	for i in range(len(files)):
		files[i] = d + files[i]
	return files
	
def test_output(file_name, cont):
	Keqing = ''
	f = open(file_name, 'w')
	for i in cont:
		Keqing = Keqing + i + ' ' + str(cont[i]) + '\n'
	f.write(Keqing)
	f.close()

### main action part ###
targetdir0 = r"./opcode/0/"	# ransomwares
targetdir1 = r"./opcode/1/"	# good files

files0 = directory_mapping(targetdir0, os.listdir(targetdir0))	# file name list of 0
files1 = directory_mapping(targetdir1, os.listdir(targetdir1))	# file name list of 1

def gen_sequence(file_name, ngram = 4):
# generate sequences and number of them of each file
# return : {'sequence' : cnt} of input file
	f = open(file_name, 'r')
	lines = f.readlines()
	l = len(lines)
	res = dict()
	if l < ngram:
		return res
	i = 0
	total = 0
	while i + ngram <= l:	# n-gram sequence modeling
		a = ''
		j = i + ngram
		while i < j:
			a = a + instr_code[lines[i]] + ' '
			i += 1
		if a not in res:
			res[a] = 1
		else:
			res[a] += 1
		total += 1
	f.close()
	for i in res:
		res[i] /= total		# estimate TF
	return res
	
def sequence_analysis(file_list, ngram = 4):
	res = dict()
	for i in file_list:
		res[i] = gen_sequence(i, ngram)		# i-th file has sequences and their TF respectively
	return res
	
# construct n-gram sequences and their TFs of input files
R_sq = sequence_analysis(files0)
N_sq = sequence_analysis(files1)

def est_sequence(analyzed_file_list):	# 해당 파일 그룹에 나타나는 시퀀스들의 idf
	seq_cnt = dict()
	tdf = dict()
	num_of_files = len(analyzed_file_list)
	for r in analyzed_file_list:
		now_file = analyzed_file_list[r]
		for s in now_file:	# s is a sequence of each analyzed_file_list
			if s not in seq_cnt:
				seq_cnt[s] = 1
			else:
				seq_cnt[s] += 1
	for d in analyzed_file_list:
		now_file = analyzed_file_list[d]
		for s in now_file:
			tf = now_file[s]
			if s not in tdf:
				tdf[s] = 0
			tdf[s] += tf * np.log(num_of_files / seq_cnt[s]) / num_of_files # estimate mean
	return tdf
	
# to find moderate value of threshold
est_R_train = est_sequence(R_sq)
est_N_train = est_sequence(N_sq)

def test_print(cont, threshold = 0.0003, k = 10):
    for i in cont:
        if k < 1:
            break
        if cont[i] > threshold:
            print(i + ' ' + str(cont[i]))
            k -= 1

# if no option, print 10 lines in default
# this helps to determine the threshold
print('est_R_train')
test_print(est_R_train)
print('est_N_train')
test_print(est_N_train)

# elect avaliable ransomware sequence
# it can also gain available normal sequence if input container with changing position
def elect_rns_seq(rns_seq_cnt, nrm_seq_cnt, threshold = 0.0003):
    res = dict()
    for r in rns_seq_cnt:
        if r not in nrm_seq_cnt and rns_seq_cnt[r] > threshold:
            res[r] = rns_seq_cnt[r]
        elif r in nrm_seq_cnt and rns_seq_cnt[r] > nrm_seq_cnt[r] + threshold:
            res[r] = rns_seq_cnt[r]
    return res

# concat available ransomware code sequence and normal code sequence
def gen_features(elected_seq1, elected_seq2):
    features = list(elected_seq1.keys()) + list(elected_seq2.keys()) # len(featurs) is len(seq1) + len(seq2)
    
    return features
    
# generate vector with analyzed files into features
def gen_vector_file(analyzed_file_list, features):
    D = len(features)
    N = len(analyzed_file_list)
    Xs = []
    for i in analyzed_file_list:
        X = []
        ith_sqs = analyzed_file_list[i]
        for j in features:
            if j in ith_sqs:
                X.append(ith_sqs[j])
            else:
                X.append(0.0)
        X = np.array(X)
        Xs.append(X)
    Xs = np.array(Xs)
    return Xs
    
# measure the accuracy of test files with generated vectors and SVM
def model_svm_accuracy(R_train_sq, R_test_sq, N_train_sq, N_test_sq, threshold = 0.0003, print_op = False):
    # available ransomware sequences and normal sequences
    avl_R_sq = elect_rns_seq(est_R_train, est_N_train, threshold = threshold)
    #avl_N_sq = dict()
    avl_N_sq = elect_rns_seq(est_N_train, est_R_train, threshold = threshold)
    
    if print_op:
        print('avl_R_sq' + ' number of sequences:' + str(len(avl_R_sq)))
        # test_print(avl_R_sq)
        print('avl_N_sq' + ' number of sequences:' + str(len(avl_N_sq)))
        # test_print(avl_N_sq)
    
    features = gen_features(avl_R_sq, avl_N_sq)
    # features = gen_features(avl_R_sq, dict())
    
    # generate train vectors with features estimated
    # features mean the secquences of the values to compare
    RX_train = gen_vector_file(R_train_sq, features)
    RX_test = gen_vector_file(R_test_sq, features)
    NX_train = gen_vector_file(N_train_sq, features)
    NX_test = gen_vector_file(N_test_sq, features)
    
    yR_train = np.zeros(RX_train.shape[0])
    yR_test = np.zeros(RX_test.shape[0])
    yN_train = np.ones(NX_train.shape[0])
    yN_test = np.ones(NX_test.shape[0])
    
    if print_op:
        # check vectors' shape
        print('RX_train\'s shape:' + str(RX_train.shape) + ', yR_train:' + str(yR_train.shape))
        print('NX_train\'s shape:' + str(NX_train.shape) + ', yNtrain:' + str(yN_train.shape))
        print('')
    
    # use Support Vector Machine
    clf = svm.SVC()
    train_set = RX_train.tolist() + NX_train.tolist()
    y_train = yR_train.tolist() + yN_train.tolist()
    clf.fit(train_set, y_train)
    
    # predict and measure the accuracy
    pred_R = clf.score(RX_test, yR_test)
    pred_N = clf.score(NX_test, yN_test)

    print('accuracy of ransomware and normal files')
    print('R:' + str(pred_R) + '\nN:' + str(pred_N))
    
# split sequence list as training list and test list
def tt_split(sq_list, train_list, test_list):
    train_ = dict()
    test_ = dict()
    for i in train_list:
        train_[i] = sq_list[i]
    for i in test_list:
        test_[i] = sq_list[i]
    return train_, test_
    
# make train-test set with split size of files and shuffle random seed
def gen_tt_set(est_R_train, est_N_train, files0, files1, R_sq, N_sq, test_sz = 0.33, rand_st = 42):
    R_train, R_test = train_test_split(files0, test_size = test_sz, random_state = rand_st)
    N_train, N_test = train_test_split(files1, test_size = test_sz, random_state = rand_st)
    
    # split with each cases
    R_train_sq, R_test_sq = tt_split(R_sq, R_train, R_test)
    N_train_sq, N_test_sq = tt_split(N_sq, N_train, N_test)
    
    return R_train_sq, R_test_sq, N_train_sq, N_test_sq
    
# generate pre factors
tst_sz_list = [0.33, 0.5, 0.66]
rnd_st_list = [7, 24, 42]

tt_sets = []
for ts in tst_sz_list:
    for rs in rnd_st_list:
        R_train_sq, R_test_sq, N_train_sq, N_test_sq = gen_tt_set(est_R_train, est_N_train, files0, files1, R_sq, N_sq, test_sz = ts, rand_st = rs)
        tt_sets.append((R_train_sq, R_test_sq, N_train_sq, N_test_sq, ts, rs))
        
# sampling and test
th_list = [0.0001, 0.0003, 0.0005]

# several test
for th in th_list:
    for i in range(len(tt_sets)):
        model_svm_accuracy(tt_sets[i][0], tt_sets[i][1], tt_sets[i][2], tt_sets[i][3], threshold = th, print_op = True)
        print('th:'+str(th)+', test_sz:'+str(tt_sets[i][4])+', rand_st:'+str(tt_sets[i][5]))
        print('')
