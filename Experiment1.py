import time
import copy
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from KAN.batchNorm import KAN_norm as KAN

def select_seed(k):
    torch.manual_seed(k)
    np.random.seed(k)
    random.seed(k)

class AR_dataset(Dataset):
    def __init__(self, 
                 dataset:pd.DataFrame, 
                 endogenous : str, 
                 pred_horizon : int = 10, 
                 exogenous : dict = {}, 
                 train_split : float = .8, 
                 test_split : float =.5, 
                 mode : str ="train", 
                 normalization : list = [], 
                 standarization : list = []):
        """
        Params:
        - csv_file: path for the document.
        - endogenous: tuple (variable:str, previous_values:int), i.e. ("a", 7): prediction for variable a using the previous 7 values.
        - pred_horizon: horizon of forecasting.
        - x_pred: dictionary of tuples that depicts the variables used for forecasting, i.e., {"a":(7, 10)}: variable "a" observations 7-10 times before of prediction.
        - train_split: value between 0, 1 that depicts the ammount of data used for training. 
        - test_split: value between 0, 1 that depicts the ammount of test_data used for testing.
        - mode: train, test or val. Depicts the mode used in the dataset. 
        - normalization: list of variables to be normalized. 
        - standarization: list of variables to be standarized. 
        """


        # Splits administration
        self.train_split = train_split
        self.test_split = test_split
        # Normalization values
        self.x_max = {}
        self.x_min = {}
        self.x_mean = {}
        self.x_std = {}
        # Horizons:
        self.pred_horizon = pred_horizon
        self.endogenous = endogenous
        self.x_variables = exogenous

        # Load data
        self.data = dataset.copy()

        # Normalization
        for variable in normalization:
            self.normalization(variable)
        for variable in standarization:
            self.standarization(variable)

        #print(f"\tTotal Len.:", len(self.data)-(self.pred_horizon + max([fin-ini for ini,fin in self.x_variables.values()])))
        if train_split < 1:
            self.change_mode(mode)
        else:
            self.dataset = self.data

    def change_mode(self, mode="train"):
        variables_used = max([fin-ini for ini,fin in self.x_variables.values()])
        len_usable_data = len(self.data) - self.pred_horizon - variables_used

        if mode == "train":
            train_len = round(len_usable_data*self.train_split) + self.pred_horizon + variables_used
            self.dataset = self.data[:train_len]
        else:
            test_ini = round(len_usable_data*self.train_split)
            self.dataset = self.data[test_ini:]
            len_usable_data = len(self.dataset) - self.pred_horizon - variables_used
            if mode == "test":
                test_len = round(len_usable_data*self.test_split) + self.pred_horizon + variables_used
                self.dataset = self.dataset[:test_len]
            elif mode == "val":
                self.dataset = self.dataset[round(len_usable_data*self.test_split):]
            else:
                raise ValueError(f"The data mode can only be train, test or val.")

        #print(f"\t{mode} Len.: ", len(self), "\n", 50*"-")
        
    def normalization(self, variable):
        # Normalization
        #print(f"Normalization of variable {variable}.")
        self.x_min[variable] = np.min(np.array(self.data[variable])[:len(self.data)*round(self.train_split)])
        self.x_max[variable] = np.max(np.array(self.data[variable])[:len(self.data)*round(self.train_split)])
        self.data[variable] = (self.data[variable]-self.x_min[variable]) / (self.x_max[variable]-self.x_min[variable])

    def standarization(self, variable):
        # Standarization
        print(f"Standarization of variable {variable}.")
        self.x_mean = np.mean(np.array(self.data[variable])[:len(self.data)*round(self.train_split)])
        self.x_std = np.std(np.array(self.data[variable])[:len(self.data)*round(self.train_split)])
        self.data[variable] = (self.data[variable]-self.x_mean[variable])/self.x_std[variable]
    
    def __len__(self):
        return len(self.dataset)-(self.pred_horizon+max([fin-ini for ini,fin in self.x_variables.values()]))
    
    def __getitem__(self, idx):
        returnable = {}
        if idx > len(self)-1:
            raise IndexError("Pallets_dataset index out of range")
        
        # X_values:
        
        for variable in self.x_variables.keys():
            index = idx+max([fin-ini for ini,fin in self.x_variables.values()])
            returnable[variable] = torch.Tensor(list(self.dataset[variable]))[index-self.x_variables[variable][1]:index-self.x_variables[variable][0]]
            

        if type(self.endogenous) == str:
            return  {"x": torch.cat([returnable[variable] for variable in self.x_variables.keys()]),
                    "y": torch.Tensor(list(self.dataset[self.endogenous]))[index : index+self.pred_horizon]}
        elif type(self.endogenous) == list:
            return  {"x": torch.cat([returnable[variable] for variable in self.x_variables.keys()]),
                    "y": torch.cat([torch.Tensor(list(self.dataset[endog]))[index : index+self.pred_horizon] for endog in self.endogenous])} 
    
    def denormalize(self, array:torch.Tensor, variable)->torch.Tensor:
        if type(variable) == list:
            values = []
            for var in range(len(variable)):
                #print(array[:, var*self.pred_horizon:var*self.pred_horizon+self.pred_horizon].shape)
                values.append(array[:, var*self.pred_horizon:var*self.pred_horizon+self.pred_horizon]*(self.x_max[variable[var]]-self.x_min[variable[var]]) + self.x_min[variable[var]])
            #print(array[:, var*self.pred_horizon:var*self.pred_horizon+self.pred_horizon]*(self.x_max[variable[var]]-self.x_min[variable[var]])+ self.x_min[variable[var]])
            values = torch.cat(values, axis=1)
            return torch.Tensor(values)
            
        return array*(self.x_max[variable]-self.x_min[variable]) + self.x_min[variable]

    def destandarize(self, array:torch.Tensor, variable)->torch.Tensor:
        return array*self.x_std[variable]+self.x_mean[variable]
     
class MA_dataset(Dataset):
    def __init__(self, raw_dataloader, KAR_model, endog, pred_horizon, mode="train", train_split=.8, test_split=.5):
        '''
        Params:
        - raw_dataloader: Logifruit_KAN_dataloader
        - KAR_model: KAR model
        - prev_values: residual values to be taken into account
        - exogenous: predicting variables
        '''
        self.endog = endog
        self.pred_horizon = pred_horizon
        self.train_split = train_split
        self.test_split = test_split
        device = "cuda" if torch.cuda.is_available() else "cpu"
        KAR_model.eval()
        out = []
        real = []
        with torch.no_grad():
            for batch in raw_dataloader:
                real.append(batch["y"])
                x = batch["x"].to(device)
                #  Forward
                outputs = KAR_model(x)
                out.append(outputs)

        pred = torch.cat(out, 0)
        real = torch.cat(real, 0)
        pred = pred.to("cpu")
        self.residuals = real-pred
        self.x = self.residuals
        self.prev_values = int(self.residuals.shape[1]/self.endog)
        self.change_mode(mode)
        self.X = self.residuals[self.interval[0] : self.interval[1]+(2*self.prev_values-1), :]
        self.Y = self.residuals[self.interval[0]+self.prev_values : self.interval[1]+(2*self.prev_values-1+self.pred_horizon), :]

    def change_mode(self, mode="train"):
        usable_len = self.residuals.shape[0] - (2*self.prev_values-1+self.pred_horizon)
        train_interval = (0, round(usable_len*self.train_split))
        test_interval = (train_interval[1], round((usable_len-train_interval[1])*self.test_split))
        val_interval = (test_interval[1], usable_len)
        if mode == "train":
            self.interval = train_interval
        elif mode == "test":
            self.interval = test_interval
        elif mode == "val":
            self.interval = val_interval
        else:
            raise ValueError(f"The data mode can only be train, test or val.")
        
    def __len__(self):
        return self.X.shape[0] - (2*self.prev_values-1)
    
    def preprocess_residuals(self, residual:torch.Tensor, values = None):
        #batch, var, prev_values = residual.shape
        residual = residual.reshape(residual.shape[0], self.endog, self.prev_values)

        if values ==None:
            values = residual.shape[2]

        aux = []
        for var in range(residual.shape[1]):
            aux2 = []
            for row in range(values):
                aux2.append(torch.sum(torch.Tensor([residual[row+col, var, -col-1] for col in range(residual.shape[2])])))
            aux.extend(aux2)
        return aux
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        
        x = self.preprocess_residuals(self.X[idx:idx+(2*self.prev_values-1)])
        y = self.preprocess_residuals(self.Y[idx:idx+(3*self.prev_values-1-self.pred_horizon)], self.prev_values)
        
        return {"x" : torch.Tensor(x), "y": torch.Tensor(y)}
     
class LSTM(nn.Module):
  def __init__(self, 
               in_features:int, 
               out_features:int, 
               **kwargs):
    super().__init__()
    self.out = torch.nn.LSTM(in_features, out_features)
  
  def forward(self, X):
    X = self.out(X)[0]
    return X

class GRU(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 **kwargs):
        super().__init__()
        self.out = torch.nn.GRU(in_features, out_features)
    
    def forward(self, X):
        return self.out(X)[0]
    
class Elman(nn.Module):
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 **kwargs):
        super().__init__()
        self.out = torch.nn.RNN(in_features, out_features)

    def forward(self, X):
        return self.out(X)[0]
        
  
class Experiment:
    def __init__(self, 
                 name: str,
                 endogenous:str,
                 AR_model: str,
                 MA_model: str, 
                 data_path: str,
                 pred_horizon:int,
                 context : dict, 
                 errors_context : int,
                 seed : int = 1812,
                 **kwargs
                 ):
        '''
        Parameters:
        - name: experiment name
        - endogenous: variable to be predicted
        - AR_model: Only accepts LSTM or KAN
        - MA_model: only accepts LSTM or KAN
        - data_path: path to data
        - pred_horizon: ammount of data to be predicted
        '''
        select_seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = name
        self.endogenous = endogenous
        self.network = {"AR": AR_model, "MA":MA_model}
        self.context = context
        self.errors_context = errors_context
        self.pred_horizon = pred_horizon
        self.path = data_path
        self.prepare_models(**kwargs)

    def load_data(self, **kwargs):
        self.data = pd.read_csv(self.path)

        self.control = AR_dataset(self.data, 
                                  endogenous=self.endogenous,
                                  exogenous = self.context, 
                                  pred_horizon = self.pred_horizon,
                                  normalization=list(self.context.keys()))
        test = AR_dataset(self.data,
                          endogenous = self.endogenous, 
                          exogenous = self.context,
                          pred_horizon = self.pred_horizon,
                          normalization=list(self.context.keys()), 
                          mode="test")
        val = AR_dataset(self.data, 
                         endogenous=self.endogenous,
                        exogenous = self.context,
                        pred_horizon = self.pred_horizon,
                        normalization=list(self.context.keys()), 
                        mode="val")
        self.train_dataloader = DataLoader(self.control, batch_size=16, shuffle=False)
        self.test_dataloader = DataLoader(test, batch_size=16, shuffle=False)
        self.val_dataloader = DataLoader(val, batch_size=16, shuffle=False)

    def load_errors(self, **kwargs):
        data = AR_dataset(self.data, 
                        endogenous=self.endogenous,
                        exogenous = self.context, 
                        pred_horizon = self.pred_horizon,
                        normalization=list(self.context.keys()),
                        train_split = 1)
        dataloader = DataLoader(data, batch_size=16, shuffle=False)

        train_residuals = MA_dataset(dataloader, self.model_AR, len(self.endogenous), self.pred_horizon)
        test_residuals = copy.deepcopy(train_residuals)
        test_residuals.change_mode("test")
        val_residuals = copy.deepcopy(train_residuals)
        val_residuals.change_mode("val")
        
        self.train_dataloader = DataLoader(train_residuals, batch_size=16, shuffle=False)
        self.test_dataloader = DataLoader(test_residuals, batch_size=16, shuffle=False)
        self.val_dataloader = DataLoader(val_residuals, batch_size=16, shuffle=False)

    def prepare_models(self, **kwargs):
        context_len = np.sum([np.abs(i)+np.abs(j) for (i, j) in self.context.values()])

        if type(self.endogenous) == list:
            pred_horizon = self.pred_horizon*len(self.endogenous)
        else:
            pred_horizon = self.pred_horizon

        print(f"Results for models {self.network['AR']}_AR and {self.network['MA']}_MA.")

        if self.network["AR"] == "LSTM":
            self.model_AR = LSTM(context_len, pred_horizon)
        elif self.network["AR"] == "KAN":
            self.model_AR = KAN(context_len, pred_horizon, [context_len*2+1])
        elif self.network["AR"] == "GRU":
            self.model_AR = GRU(context_len, pred_horizon)
        elif self.network["AR"] == "Elman":
            self.model_AR = Elman(context_len, pred_horizon)
        else:
            raise ValueError(f"{self.network['AR']} is not implemented for AR model.")
        
        if self.network["MA"] == "LSTM":
                self.model_MA = LSTM(self.errors_context*len(self.endogenous), pred_horizon)
        elif self.network["MA"] == "KAN":
            self.model_MA = KAN(self.errors_context*len(self.endogenous), pred_horizon, [self.errors_context*len(self.endogenous)*2+1])
        elif self.network["MA"] == "GRU":
            self.model_MA = GRU(self.errors_context*len(self.endogenous), pred_horizon)
        elif self.network["MA"] == "Elman":
            self.model_MA = Elman(self.errors_context*len(self.endogenous), pred_horizon)
        else:
            raise ValueError(f"{self.network['MA']} is not implemented for MA model.")
        
    def load_models(self,
                    AR_path:str = None,
                    MA_path:str = None):
        if AR_path != None:
            self.model_AR.load_state_dic(torch.load(AR_path))
        if MA_path != None:
            self.model_MA.load_state_dic(torch.load(MA_path))

    def autoregression(self, **kwargs):
        self.load_data()
        
        print("\n---- Model for autorregression ----")
        self.model_AR = self.model_AR.to(self.device)
        start = time.time()
        self.train(self.model_AR, "AR", **kwargs)
        end = time.time()
        print(f"Time required {end-start}.")
        self.test(self.model_AR, "AR", **kwargs)

    def error_regression(self, **kwargs):
        self.load_errors()
        
        print("\n---- Model for error regression ----")
        self.model_MA = self.model_MA.to(self.device)
        start = time.time()
        self.train(self.model_MA, "MA", **kwargs)
        end = time.time()
        print(f"Time required {end-start}.")
        self.test(self.model_MA, "MA", **kwargs)

    def train(self, 
              model, 
              mode : str,
              criterion = nn.MSELoss(), 
              patience = 20, 
              epochs = 1000, 
              scheduler_patience = 5):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=scheduler_patience, min_lr=1e-16)

        best_test_RMSE = np.inf
        best_epoch = 0
        contador = 0

        print("\n---- Start Training ----")
        best_test_RMSE = np.inf
        best_epoch = 0
        contador = 0

        for epoch in range(epochs):
            # TRAIN NETWORK
            model.train()
            labels = []
            train_loss = 0
            predicts = []
            for batch in self.train_dataloader:
                #ids = batch["idx"].to('cpu').numpy()

                # zero the parameter gradients
                optimizer.zero_grad()

                #  Forward
                #return net(x=x, y=labels, criterion=criterion)
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                outputs = model(x)
                loss = criterion(y, outputs)
                loss.backward()
                optimizer.step()

                # print statistics
                train_loss += loss.item()
                prediction = self.control.denormalize(outputs, self.endogenous)
                predicts.extend(prediction.tolist())
                real = self.control.denormalize(batch["y"], self.endogenous)
                labels.extend(real.tolist())

            labels = torch.Tensor(labels)
            train_RMSE = (torch.sum((labels - torch.Tensor(predicts))**2)/(labels.shape[0]*labels.shape[1]))**(1/2)
            train_loss /= len(self.train_dataloader)
            # TEST NETWORK
            model.eval()
            with torch.no_grad():
                predicts = []
                labels = []
                test_loss = 0
                for batch in self.test_dataloader:
                    #ids = batch["idx"].to('cpu').numpy()
                    #  Forward
                    x = batch["x"].to(self.device)
                    y = batch["y"].to(self.device)
                    outputs = model(x)
                    loss = criterion(y, outputs)
                    test_loss += loss.item()
                    prediction = self.control.denormalize(outputs, self.endogenous)
                    predicts.extend(prediction.tolist())
                    real = self.control.denormalize(batch["y"], self.endogenous)
                    labels.extend(real.tolist())
            
            labels = torch.Tensor(labels)
            test_RMSE = (torch.sum((labels - torch.Tensor(predicts))**2)/(labels.shape[0]*labels.shape[1]))**(1/2)
            test_loss /= len(self.test_dataloader)
            scheduler.step(train_loss)

            '''print("[Epoch {}] Train Loss: {:.6f} - Train RMSE: {:.2f} - Test Loss {:.6f} - Test RMSE: {:.2f}".format(
                epoch + 1, train_loss, train_RMSE, test_loss, test_RMSE
            ))'''

            if  test_RMSE < best_test_RMSE:
                best_epoch = epoch+1
                best_test_RMSE = test_RMSE
                best_train_RMSE = train_RMSE
                torch.save(model.state_dict(), f"../Models/{self.name}_{mode}.pt")
                contador = 0
            else:
                contador += 1

            if contador == patience:
                break
        
        print("Train accuracy", best_train_RMSE, "in epoch", best_epoch)
        print("Test accuracy", best_test_RMSE, "in epoch", best_epoch)

    def test(self, 
             model, 
             mode:str,
             **kwargs):
        model.load_state_dict(torch.load(f"../Models/{self.name}_{mode}.pt"))
        model.eval()
        out = []
        real = []
        with torch.no_grad():
            for batch in self.val_dataloader:
                x = batch["x"].to(self.device)
                real.append(batch["y"])
                #  Forward
                outputs = model(x)
                out.append(outputs)

        output = torch.cat(out, 0)
        real = torch.cat(real, 0)

        output = self.control.denormalize(output, self.endogenous)
        y = self.control.denormalize(real, self.endogenous)
        pred = output.to("cpu")

        y = torch.Tensor(y)
        pred = torch.Tensor(pred)
        rmse_test = (torch.sum((y - pred)**2)/(y.shape[0]*y.shape[1]))**(1/2)
        print(f"Validation RMSE: {rmse_test}")

    def fit(self, **kwargs):
        self.autoregression(**kwargs)
        self.error_regression(**kwargs)
