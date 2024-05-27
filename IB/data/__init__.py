from scipy import io
from sklearn.model_selection import train_test_split
import numpy as np
import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def load(dataset_name):
    if dataset_name not in {'SYN','MNIST','CIFAR','fashion_MNIST_combined_labels'}:
        raise Exception("Unknown data set: '"+dataset_name+"'")
    dataset = {
        'SYN':_load_tishby_data,
        'MNIST':_load_MNIST,
        'CIFAR':_load_CIFAR,
        'fashion_MNIST': _load_fashion_MNIST,
        'fashion_MNIST_combined_labels': _load_fashion_MNIST_combined_labels
    }[dataset_name]
    print(dataset_name)
    return dataset()
def load_split(dataset_name):
    if dataset_name not in {'MNIST','CIFAR','fashion_MNIST','fashion_MNIST_combined_labels'}:
        raise Exception("Unknown data set: '"+dataset_name+"'")
    dataset = {
        'MNIST':_load_MNIST_split,
        'CIFAR':_load_CIFAR_split,
        'fashion_MNIST':_load_fashion_MNIST_split,
        'fashion_MNIST_combined_labels':_load_fashion_MNIST_split_combined_labels
    }[dataset_name]
    return dataset()

# Returns (X_train, X_test, y_train, y_test)
def split(X,y,test_frac,seed=None):
    return train_test_split(X, y, random_state=seed, test_size=test_frac, shuffle=True, stratify=y)

# Load data from Tishby paper
def _load_tishby_data():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    # Load data as is
    data = io.loadmat(os.path.join(__location__, 'var_u.mat')) # OBS loads in a weird JSON
    X = data["F"] # (4096, 12)
    y = data["y"] # (1, 4096)
    y = y.squeeze()
    
    return X,y

### CIFAR
def _load_CIFAR():
    X1,y1,X2,y2 = _load_CIFAR_split()
    return np.concatenate((X1,X2),axis=0), np.concatenate((y1,y2),axis=0)
def _load_CIFAR_split():
    import pickle
    Xs,ys = [],[]
    for f in ["data_batch_"+str(i) for i in range(1,6)]:
        with open("data/cifar/"+f, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        Xs.append(data[b'data'])
        ys.append(data[b'labels'])
    X_train = np.concatenate(Xs).reshape((-1,32,32,3))
    y_train = np.concatenate(ys)
    
    with open("data/cifar/test_batch",'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    X_test = data[b'data'].reshape((-1,32,32,3))
    y_test = data[b'labels']
    
    return X_train, X_test, y_train, y_test

### MNIST
def _load_MNIST():
    X1,y1,X2,y2 = _load_MNIST_split()
    return np.concatenate((X1,X2),axis=0), np.concatenate((y1,y2),axis=0)
def _load_MNIST_split():
    path = "data/mnist/mnist.data"
    X_train, y_train = _read_idx_file(path, 28*28)
    X_test, y_test   = _read_idx_file(path+".t", 28*28)
    X_train, X_test  = X_train.reshape((-1,28,28,1)), X_test.reshape((-1,28,28,1))
    return X_train, X_test, y_train, y_test
def _read_idx_file(path, d, sep=None):
    X = []
    Y = []
    with open(path) as f:
        for l in f:
            x = np.zeros(d)
            l = l.strip().split() if sep is None else l.strip().split(sep)
            Y.append(int(l[0]))
            for pair in l[1:]:
                pair = pair.strip()
                if pair=='':
                    continue
                (i,v) = pair.split(":")
                if v=='':
                    import pdb; pdb.set_trace()
                x[int(i)-1] = float(v)
            X.append(x)
    return np.array(X),np.array(Y)

def _load_fashion_MNIST():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train.reshape((-1, 28, 28, 1)), X_test.reshape((-1, 28, 28, 1))
    return np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0)

def _load_fashion_MNIST_split():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train.reshape((-1, 28, 28, 1)), X_test.reshape((-1, 28, 28, 1))
    return X_train, X_test, y_train, y_test


# Load the Fashion MNIST dataset with combined labels
def _load_fashion_MNIST_combined_labels():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train.reshape((-1, 28, 28, 1)), X_test.reshape((-1, 28, 28, 1))
    y_train_combined, y_test_combined = _combine_labels(y_train, y_test)
    return np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train_combined, y_test_combined), axis=0)

# Load the Fashion MNIST dataset and split it with combined labels
def _load_fashion_MNIST_split_combined_labels():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train.reshape((-1, 28, 28, 1)), X_test.reshape((-1, 28, 28, 1))
    y_train_combined, y_test_combined = _combine_labels(y_train, y_test)
    return X_train, X_test, y_train_combined, y_test_combined

# Function to combine similar labels
def _combine_labels(y_train, y_test):
    # Define the label mapping:
    # - Combining t-shirt/top (0) and shirt (6) into label 0
    # - Trouser (1) remains as label 1
    # - Combining pullover (2) and coat (4) into label 2
    # - Dress (3) remains as label 3
    # - Bag (8) to label 4
    # - Combining sandal (5), sneaker (7), and ankle boot (9) into label 5
    label_map = {0: 0, 6: 0, 1: 1, 2: 2, 4: 2, 3: 3, 8: 4, 5: 5, 7: 5, 9: 5}
    y_train_combined = np.copy(y_train)
    y_test_combined = np.copy(y_test)
    
    for original_label, new_label in label_map.items():
        y_train_combined[y_train == original_label] = new_label
        y_test_combined[y_test == original_label] = new_label
    
    return y_train_combined, y_test_combined

# Load combined datasets
X, y = _load_fashion_MNIST_combined_labels()
X_train, X_test, y_train, y_test = _load_fashion_MNIST_split_combined_labels()

# Print the unique labels to verify
print(f'Combined training labels: {np.unique(y_train)}')
print(f'Combined test labels: {np.unique(y_test)}')

# Print some examples to see the changes
print("Example of original and combined labels:")
for i in range(10):
    print(f'Combined: {y_train[i]}')