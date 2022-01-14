# Static analysis for detecting malware instruction

## Requirements
1. opcode sequence for static analysis
2. API sequence for dynamic analysis  
Suppose the sequence data was given. Design a model that classifies malware code and normal code from them, and evaluate its accuracy.  
In this project, the first method, static analysis was applied.

## Environment
- OS: Ubuntu Linux 20.04 LTS
- Language: Python 3.8.5
- About 3GB free Memory required

## Note:
- 1~5.txt are sample TFs of several files.
- result data sample are test[1-10].txt files.
- final best output is "test6.txt"
- and please compare it with "test8.txt"
- I recommend to check this subject via running with "Sampling.ipynb."


## How to run
Assume that current directory is on `./src`  
- Python: `$python3 practice2.py`
- Jupyter Notebook: `$jupyter-notebook -ip=0.0.0.0 --port=8888 --allow-root` or just execute ipynb file.

## Main algorithm
- SVM(Support Vector Machine)
- N-gram
- TF-IDF