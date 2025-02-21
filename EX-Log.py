from Experiment1 import Experiment
import sys

dataset = "Logifruit"
pred = int(sys.argv[1])
AR_model = sys.argv[2]
MA_model = sys.argv[3]

endog = ["production"]

exog = ["demand", "deliveries", "National_holiday", "Provintial_holiday"]

fourier = ["National_holiday", "Provintial_holiday", 'fourier_sin_order1', 'fourier_cos_order1']




X = {var:(0, 14) for var in endog}
X.update({var:(0, 7) for var in exog})
X.update({var:(-pred, 0) for var in fourier})

model = Experiment(name=f"{dataset}/{pred}/{AR_model}-{MA_model}", 
                   endogenous = endog,
                   AR_model=AR_model,
                   MA_model=MA_model,
                   data_path="../../00-Data/"+dataset+".csv", 
                   pred_horizon=pred, 
                   context=X,
                   errors_context=pred,
                   batch_size=30)
model.fit()
