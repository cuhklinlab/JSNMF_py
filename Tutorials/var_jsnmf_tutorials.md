

## 1. JSNMF same H model

Initialize the model with two Pytorch Tensor and true labels.
```
from JSNMF.model import JSNMF_same_H
test_model = JSNMF_same_H(X1,X2,label)
```
After initializing, run the model is also quite easy: 
```
test_model.run()
```
The result is saved in `test_model.result`, which is a dict, the complete graph S, can be get with easy access:
```
S = test_model.result['S']
```



## 2. JSNMF for data that has 3 modalities

```
from JSNMF.model import JSNMF_3mod
test_model = JSNMF_same_H(X1,X2,X3,label)
```
After initializing, run the model is also quite easy: 
```
test_model.run()
```
The result is saved in `test_model.result`, which is a dict, and the complete graph S, can be get with easy access:
```
S = test_model.result['S']
```



## 3. JSNMF for batch correction of 5  Paired-Tag datasets 

Load data
```
import torch
import numpy as np
from JSNMF.model import JSNMF_be5
import scipy.io

11 = torch.from_numpy(data['X11'].toarray())
X12 = torch.from_numpy(data['X12'].toarray())
X21 = torch.from_numpy(data['X21'].toarray())
X23 = torch.from_numpy(data['X23'].toarray())
X31 = torch.from_numpy(data['X31'].toarray())
X34 = torch.from_numpy(data['X34'].toarray())
X41 = torch.from_numpy(data['X41'].toarray())
X45 = torch.from_numpy(data['X45'].toarray())
X51 = torch.from_numpy(data['X51'].toarray())
X56 = torch.from_numpy(data['X56'].toarray())
num_c = []
num_c.append(data['num_clu1'][0][0])
num_c.append(data['num_clu2'][0][0])
num_c.append(data['num_clu3'][0][0])
num_c.append(data['num_clu4'][0][0])
num_c.append(data['num_clu5'][0][0])

```
Initialize model 
```
model = JSNMF_be5(X11, X12, X21, X23,X31,X34,X41, X45,X51,X56,num_c)

```
Run the model
```
model.run()
```
The result is saved in `model.result`, which is a dict.







