import sys

from IB.models import load as load_model
from IB.util.estimator import BinningEstimator, QuantizedEstimator
from IB.experiment import run_experiment

if len(sys.argv)<2:
    print("Missing arguments, binning.py <experiment> [repeats=50].")
    sys.exit(1)

# Activation function
exp     = sys.argv[1]
repeats = int(sys.argv[2]) if len(sys.argv)>=3 else 50

if exp not in ("MNIST","CIFAR","SYN-Tanh","SYN-ReLU"):
    print("Experiment must be one of 'MNIST', 'CIFAR' or 'SYN-{Tanh,ReLU}'")
    sys.exit(1)

# Model
if exp[:3]=="SYN":
    _model  = load_model("SYN")
    act_fun = exp[4:].lower()
    dname   = "SYN"
    Model   = lambda: (_model(activation=act_fun, quantize=False), False)
    epochs  = 8000
else:
    _model = load_model(exp)
    dname  = exp
    Model  = lambda: (_model(quantize=False), False)
    epochs = 2000

# MI estimators
estimators = []
for n_bins in [30,100,256]:
    estimators.append(BinningEstimator("uniform", n_bins=n_bins, bounds="global"))

res_path = "out/binning/"+exp+"/"

print("Starting binning experiment.")
run_experiment(Model, estimators, dname, epochs=epochs, repeats=repeats, out_path=res_path)
