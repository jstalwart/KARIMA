from Experiment1 import Experiment
import sys

dataset = "ECL"
batch_size = int(sys.argv[1])
pred = int(sys.argv[2])
AR_model = sys.argv[3]
MA_model = sys.argv[4]



endog = [f"MT_{i+1:03}" for i in range(370//2)]
fourier = ['fourier_sin_order1', 'fourier_cos_order1']




X = {var:(0, pred//2) for var in endog}
X.update({var:(-pred, 0) for var in fourier})

model = Experiment(name=f"{dataset}/{pred}/{AR_model}-{MA_model}", 
                   endogenous = endog,
                   AR_model=AR_model,
                   MA_model=MA_model,
                   data_path="../../00-Data/"+dataset+".csv", 
                   pred_horizon=pred, 
                   context=X,
                   errors_context=pred,
                   batch_size=4)
model.fit()