from Experiment1 import Experiment
import sys

dataset = sys.argv[1]
pred = int(sys.argv[2])
AR_model = sys.argv[3]
MA_model = sys.argv[4]

endog = ['ILITOTAL Alabama', 'ILITOTAL Alaska', 'ILITOTAL Arizona', 'ILITOTAL Arkansas', 'ILITOTAL California', 'ILITOTAL Colorado', 'ILITOTAL Connecticut', 'ILITOTAL Delaware', 
       'ILITOTAL District of Columbia', 'ILITOTAL Georgia', 'ILITOTAL Hawaii', 'ILITOTAL Idaho', 'ILITOTAL Illinois', 'ILITOTAL Indiana', 'ILITOTAL Iowa', 'ILITOTAL Kansas', 
       'ILITOTAL Kentucky', 'ILITOTAL Louisiana', 'ILITOTAL Maine', 'ILITOTAL Maryland', 'ILITOTAL Massachusetts', 'ILITOTAL Michigan', 'ILITOTAL Minnesota', 'ILITOTAL Mississippi', 
       'ILITOTAL Missouri', 'ILITOTAL Montana', 'ILITOTAL Nebraska', 'ILITOTAL Nevada', 'ILITOTAL New Hampshire', 'ILITOTAL New Jersey', 'ILITOTAL New Mexico', 'ILITOTAL New York', 
       'ILITOTAL New York City', 'ILITOTAL North Carolina', 'ILITOTAL North Dakota', 'ILITOTAL Ohio', 'ILITOTAL Oklahoma', 'ILITOTAL Oregon', 'ILITOTAL Pennsylvania', 'ILITOTAL Puerto Rico', 
       'ILITOTAL Rhode Island', 'ILITOTAL South Carolina', 'ILITOTAL South Dakota', 'ILITOTAL Tennessee', 'ILITOTAL Texas', 'ILITOTAL Utah', 'ILITOTAL Vermont', 'ILITOTAL Virgin Islands', 
       'ILITOTAL Virginia', 'ILITOTAL Washington', 'ILITOTAL West Virginia', 'ILITOTAL Wisconsin', 'ILITOTAL Wyoming']

exog = ['NUM. OF PROVIDERS Alabama', 'NUM. OF PROVIDERS Alaska', 'NUM. OF PROVIDERS Arizona', 'NUM. OF PROVIDERS Arkansas', 'NUM. OF PROVIDERS California', 'NUM. OF PROVIDERS Colorado', 'NUM. OF PROVIDERS Connecticut', 'NUM. OF PROVIDERS Delaware', 'NUM. OF PROVIDERS District of Columbia', 'NUM. OF PROVIDERS Georgia', 'NUM. OF PROVIDERS Hawaii', 'NUM. OF PROVIDERS Idaho', 'NUM. OF PROVIDERS Illinois', 'NUM. OF PROVIDERS Indiana', 'NUM. OF PROVIDERS Iowa', 'NUM. OF PROVIDERS Kansas', 'NUM. OF PROVIDERS Kentucky', 'NUM. OF PROVIDERS Louisiana', 'NUM. OF PROVIDERS Maine', 'NUM. OF PROVIDERS Maryland', 'NUM. OF PROVIDERS Massachusetts', 'NUM. OF PROVIDERS Michigan', 'NUM. OF PROVIDERS Minnesota', 'NUM. OF PROVIDERS Mississippi', 'NUM. OF PROVIDERS Missouri', 'NUM. OF PROVIDERS Montana', 'NUM. OF PROVIDERS Nebraska', 'NUM. OF PROVIDERS Nevada', 'NUM. OF PROVIDERS New Hampshire', 'NUM. OF PROVIDERS New Jersey', 'NUM. OF PROVIDERS New Mexico', 'NUM. OF PROVIDERS New York', 'NUM. OF PROVIDERS New York City', 'NUM. OF PROVIDERS North Carolina', 'NUM. OF PROVIDERS North Dakota', 'NUM. OF PROVIDERS Ohio', 'NUM. OF PROVIDERS Oklahoma', 'NUM. OF PROVIDERS Oregon', 'NUM. OF PROVIDERS Pennsylvania', 'NUM. OF PROVIDERS Puerto Rico', 'NUM. OF PROVIDERS Rhode Island', 'NUM. OF PROVIDERS South Carolina', 'NUM. OF PROVIDERS South Dakota', 'NUM. OF PROVIDERS Tennessee', 'NUM. OF PROVIDERS Texas', 'NUM. OF PROVIDERS Utah', 'NUM. OF PROVIDERS Vermont', 'NUM. OF PROVIDERS Virgin Islands', 'NUM. OF PROVIDERS Virginia', 'NUM. OF PROVIDERS Washington', 'NUM. OF PROVIDERS West Virginia', 'NUM. OF PROVIDERS Wisconsin', 'NUM. OF PROVIDERS Wyoming', 'TOTAL PATIENTS Alabama', 'TOTAL PATIENTS Alaska', 'TOTAL PATIENTS Arizona', 'TOTAL PATIENTS Arkansas', 'TOTAL PATIENTS California', 'TOTAL PATIENTS Colorado', 'TOTAL PATIENTS Connecticut', 'TOTAL PATIENTS Delaware', 'TOTAL PATIENTS District of Columbia', 'TOTAL PATIENTS Georgia', 'TOTAL PATIENTS Hawaii', 'TOTAL PATIENTS Idaho', 'TOTAL PATIENTS Illinois', 'TOTAL PATIENTS Indiana', 'TOTAL PATIENTS Iowa', 'TOTAL PATIENTS Kansas', 'TOTAL PATIENTS Kentucky', 'TOTAL PATIENTS Louisiana', 'TOTAL PATIENTS Maine', 'TOTAL PATIENTS Maryland', 'TOTAL PATIENTS Massachusetts', 'TOTAL PATIENTS Michigan', 'TOTAL PATIENTS Minnesota', 'TOTAL PATIENTS Mississippi', 'TOTAL PATIENTS Missouri', 'TOTAL PATIENTS Montana', 'TOTAL PATIENTS Nebraska', 'TOTAL PATIENTS Nevada', 'TOTAL PATIENTS New Hampshire', 'TOTAL PATIENTS New Jersey', 'TOTAL PATIENTS New Mexico', 'TOTAL PATIENTS New York', 'TOTAL PATIENTS New York City', 'TOTAL PATIENTS North Carolina', 'TOTAL PATIENTS North Dakota', 'TOTAL PATIENTS Ohio', 'TOTAL PATIENTS Oklahoma', 'TOTAL PATIENTS Oregon', 'TOTAL PATIENTS Pennsylvania', 'TOTAL PATIENTS Puerto Rico', 'TOTAL PATIENTS Rhode Island', 'TOTAL PATIENTS South Carolina', 'TOTAL PATIENTS South Dakota', 'TOTAL PATIENTS Tennessee', 'TOTAL PATIENTS Texas', 'TOTAL PATIENTS Utah', 'TOTAL PATIENTS Vermont', 'TOTAL PATIENTS Virgin Islands', 'TOTAL PATIENTS Virginia', 'TOTAL PATIENTS Washington', 'TOTAL PATIENTS West Virginia', 'TOTAL PATIENTS Wisconsin', 'TOTAL PATIENTS Wyoming']
fourier = ['fourier_sin_order1', 'fourier_cos_order1', 'fourier_sin_order2', 'fourier_cos_order2']




X = {var:(0, 14) for var in endog}
X.update({var:(0, 10) for var in exog})
X.update({var:(-pred, 0) for var in fourier})

model = Experiment(name=f"{dataset}/{pred}/{AR_model}-{MA_model}", 
                   endogenous = endog,
                   AR_model=AR_model,
                   MA_model=MA_model,
                   data_path="../../00-Data/"+dataset+".csv", 
                   pred_horizon=pred, 
                   context=X,
                   errors_context=pred)
model.fit()