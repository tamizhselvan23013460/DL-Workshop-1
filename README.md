# DL-Workshop-1

## Binary Classification with Neural Networks on the Census Income Dataset

## ALGORITHMS:
### Step 1 : 

Import necessary libraries such as Torch, NumPy, Pandas, and Matplotlib for model building.

### Step 2 :

Load the “income.csv” dataset using Pandas and display its structure and basic information.

### Step 3 :

Define categorical, continuous, and label columns for model input and output preparation.

### Step 4 :

Convert categorical columns to category datatype for embedding processing in neural network.

### Step 5 :

Handle missing values by adding placeholder categories or filling continuous values with mean.

### Step 6 :

Encode categorical columns as numerical category codes and convert to PyTorch integer tensors.

### Step 7 :

Convert continuous columns to float tensors after normalization and missing value imputation.

### Step 8 :

Split dataset into training and testing tensors for categorical, continuous, and label columns.

### Step 9 :

Define the TabularModel class with embedding, dropout, normalization, and dense layer structures.

### Step 10 :

Combine categorical embeddings and continuous features, then pass through hidden layers sequentially.

### Step 11 :

Initialize model, define cross-entropy loss function, and select Adam optimizer for gradient updates.

### Step 12 :

Train model for 300 epochs, compute loss, backpropagate, and update parameters using optimizer.

### Step 13 :

Evaluate test accuracy by comparing predicted class outputs with true labels and display performance.

## PROGRAMS :
``` python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline
df = pd.read_csv('income.csv')

print(len(df))
df.head()

df['label'].value_counts()

df.columns

cat_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']
print(f'cat_cols has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col has {len(y_col)} column')

for col in cat_cols:
  df[col] = df[col].astype('category')
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)

for col in cat_cols:
  df[col] = df[col].astype('category')

# Fill NaN values in categorical columns with a placeholder
for col in cat_cols:
    if df[col].isnull().any():
        df[col] = df[col].cat.add_categories('Missing').fillna('Missing')

cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)

cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)
cats[:5]
cats = torch.tensor(cats, dtype=torch.int64)
cats

# Fill NaN values in continuous columns with the mean
for col in cont_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts[:5]
conts = torch.tensor(conts, dtype=torch.float32)
conts

df.dropna(subset=y_col, inplace=True)
y = torch.tensor(df[y_col].values, dtype=torch.int64).flatten()
b = len(df) # total records
t = 5000 # test size
cat_train = cats[:b-t]
con_train = conts[:b-t]
y_train = y[:b-t]
cat_test = cats[b-t:b-t+t]
con_test = conts[b-t:b-t+t]
y_test = y[b-t:b-t+t]
torch.manual_seed(33)

class TabularModel(nn.Module):
  def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
    # Call the parent __init__
    super().__init__()
    # Set up the embedding, dropout, and batch normalization layer attributes
    self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
    self.emb_drop = nn.Dropout(p)
    self.bn_cont = nn.BatchNorm1d(n_cont)
    # Assign a variable to hold a list of layers
    layerlist = []
    # Assign a variable to store the number of embedding and continuous layers
    n_emb = sum((nf for ni,nf in emb_szs))
    n_in = n_emb + n_cont
    # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
    for i in layers:
      layerlist.append(nn.Linear(n_in,i))
      layerlist.append(nn.ReLU(inplace=True))
      layerlist.append(nn.BatchNorm1d(i))
      layerlist.append(nn.Dropout(p))
      n_in = i
    layerlist.append(nn.Linear(layers[-1],out_sz))
    # Convert the list of layers into an attribute
    self.layers = nn.Sequential(*layerlist)
  def forward(self, x_cat, x_cont):
    # Extract embedding values from the incoming categorical data
    embeddings = []
    for i,e in enumerate(self.embeds):
      embeddings.append(e(x_cat[:,i]))
    x = torch.cat(embeddings, 1)
    # Perform an initial dropout on the embeddings
    x = self.emb_drop(x)
    # Normalize the incoming continuous data
    x_cont = self.bn_cont(x_cont)
    x = torch.cat([x, x_cont], 1)
    # Set up model layers
    x = self.layers(x)
    return x

model = TabularModel(emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)
model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import time
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
  i+=1
  y_pred = model(cat_train, con_train)
  loss = criterion(y_pred, y_train)
  losses.append(loss)
  # a neat trick to save screen space:
  if i%25 == 1:
    print(f'epoch: {i:3} loss: {loss.item():10.8f}')

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
print(f'epoch: {i:3} loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

plt.plot([loss.item() for loss in losses])
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss")
plt.show()


with torch.no_grad():
  y_val = model(cat_test, con_test)
  loss = criterion(y_val, y_test)
print(f'CE Loss: {loss:.8f}')


correct = 0
for i in range(len(y_test)):
  if y_val[i].argmax().item() == y_test[i].item():
    correct += 1
accuracy = correct / len(y_test) * 100
print(f'{correct} out of {len(y_test)} = {accuracy:.2f}% correct')


```

## OUTPUTS :
### Len and preview of datasets:

<img width="1405" height="340" alt="image" src="https://github.com/user-attachments/assets/f1fae248-b587-46ff-8934-671775c3ae82" />

### Columns:

<img width="932" height="131" alt="image" src="https://github.com/user-attachments/assets/3144cd76-f2ec-4c69-8ee2-31571cbcbca2" />

### Len of cat,cont & Y col :

<img width="326" height="122" alt="image" src="https://github.com/user-attachments/assets/fe866a13-e648-4d92-82a2-31ceb720535f" />

### View continuous columns with the mean:

<img width="416" height="257" alt="image" src="https://github.com/user-attachments/assets/6302fda8-a652-42a5-95b3-947097a6bdff" />

### Model:

<img width="1205" height="542" alt="image" src="https://github.com/user-attachments/assets/fda191a5-16b9-4b32-baa0-014ecaf57a27" />

### Loss :

<img width="400" height="463" alt="image" src="https://github.com/user-attachments/assets/0326daf1-5bf7-41e5-bbce-3447e84f4bb8" />

### Entropy Loss Vs Training Loss :

<img width="862" height="687" alt="image" src="https://github.com/user-attachments/assets/9d8c9910-e3d7-435d-b9a0-e7af8b309c97" />

### CE Loss :

<img width="515" height="70" alt="image" src="https://github.com/user-attachments/assets/7876a2fd-05eb-4eb5-a082-b1f955ad8821" />

### Accuracy :

<img width="517" height="67" alt="image" src="https://github.com/user-attachments/assets/2f358947-ac3b-481a-ac52-ed50aac541cc" />


## RESULT :

Thus , The model accurately classifies individuals’ income levels using binary classification on the Census Income dataset was executed successfully.




