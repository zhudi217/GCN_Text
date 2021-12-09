# GCN_Text

# Usage
1. Clone the git repo
```
git clone https://github.com/zhudi217/GCN_Text
```
2. Install NLTK. Download stopwords and punkt from NLTK.
```
cd GCN_Text
pip install NLTK
python3 nltk_download.py
```
3. To train pure Python version of GCN
```
python3 train.py
```
Sample output
```
[Epoch 0]: Evaluation accuracy of trained nodes: 0.0120594
[Epoch 0]: Evaluation accuracy of test nodes: 0.0450450
[Epoch 50]: Evaluation accuracy of trained nodes: 0.2133581
[Epoch 50]: Evaluation accuracy of test nodes: 0.1981982
......
```
4. To train GCN with C++ matrix multiplication kernel, change the current directory to GCN_Text/cpp_kernel first. Then compile the kernel with
```
cmake .
make
```
Train the model. Please note that the matrix multiplication kernel is too slow, it might take a while for the model to reach an evaluation checkpoint.
```
python3 train.py
```
Sample output
```

```
