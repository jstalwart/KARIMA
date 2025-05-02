from Experiment1 import Experiment
import sys

dataset = sys.argv[1]
pred = int(sys.argv[2])
AR_model = sys.argv[3]
MA_model = sys.argv[4]

endog = ['OT']

X = {"OT":(0, 72),
     "HUFL":(0, 24), 
     "HULL":(0, 24),
     "MUFL":(0, 24),
     "MULL":(0, 24),
     "LUFL":(0, 24),
     "LULL":(0, 24),
     "fourier_sin_order1":(-pred, 0),
     "fourier_cos_order1":(-pred, 0)}

model = Experiment(name=f"{dataset}/{pred}/{AR_model}-{MA_model}", 
                   endogenous = endog,
                   AR_model=AR_model,
                   MA_model=MA_model,
                   data_path="../../00-Data/"+dataset+".csv", 
                   pred_horizon=pred, 
                   context=X,
                   errors_context=pred)
model.fit()