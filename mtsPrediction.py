from mathlib import least_squares, weighted_std
import numpy as np
import scipy

from copy import deepcopy
class multiOutputRegressor:
    def __init__(self, estimator):
        self.models = []
        self.estimator = estimator

    def fit(self, X, y):
        for i in range(y.shape[1]):
            model = deepcopy(self.estimator).fit(X, y[:,i])
            self.models.append(model)
        return self
    
    def predict(self, X):
        return np.array([model.predict(X) for model in self.models]).T


class LeastSquares:
    def __init__(self, params={'lambd': 0}):
        self.params = params

    def fit(self, X, y, weights=None):
        self.x = least_squares(X, y, weights=weights, params=self.params)
        return self

    def predict(self, X):
        return X @ self.x
    
class tsEncodingFFT:
    def __init(self):
        pass

    def encode(self, ts):
        fft = np.fft.fft(ts)
        return np.append(np.real(fft), np.imag(fft))
    
    def decode(self, encoded, decodeLen):
        fft = encoded[:len(encoded)//2] + 1j * encoded[len(encoded)//2:]
        return np.fft.ifft(fft).real


class tsEncodingPoly:
    def __init__(self, m):
        self.m = m
        
    def encode(self, ts):
        t = np.arange(len(ts))
        pMod = np.polyfit(t, ts, deg=self.m)
        return pMod
    
    def decode(self, encoded, decodeLen):
        pMod =  encoded[:self.m+1]
        t = np.arange(decodeLen)
        trend = np.polyval(pMod, t)
        return trend
    
class tsEncodingNone:
    def __init__(self):
        pass

    def encode(self, ts):
        return ts
    
    def decode(self, encoded, decodeLen):
        return encoded
    
class tsEncodingDistParams:
    def __init__(self):
        pass

    def encode(self, ts):
        return np.append(np.polyfit(np.arange(ts.shape[0]), ts, deg=2), np.array([np.mean(ts), np.std(ts), np.max(ts), np.min(ts), scipy.stats.skew(ts), scipy.stats.kurtosis(ts)]))
    
    def decode(self, encoded, decodeLen):
        return None

class mtsPredictionModule:
    # train an eigen model that given the current state embedding of the system, predict the next state embedding using the eigenvalues output from the input state embedding to project the next state embedding that will yield another set of eigenvalues from the eigen model
    def __init__(self):
        self.mods = []
        self.biases = []
    
    def ndimTsEmbedding(self, X):
        # fft = np.fft.fft2(X)
        # inputvec = np.append(np.real(fft).flatten(), np.imag(fft).flatten())
        inputvec = []
        for i in range(X.shape[1]):
            ts = X[:,i]
            tsMod = self.inputEncodeFunc(ts)
            inputvec.append(tsMod)
        inputvec = np.concatenate(inputvec)
        return inputvec

    def fit(self, Xinput, Xexog, Xoutput, sig, inputEncoding, outputEncoding, horizon=1, window=1):
        # evolution on the eigenModel
        # use the optimize minimize function to find the best parameters for eigenmodel's least squares solution
        self.horizon = horizon
        self.inputEncoding = inputEncoding
        self.inputEncodeFunc = self.inputEncoding.encode
        self.inputDecodeFunc = self.inputEncoding.decode
        self.outputEncoding = outputEncoding
        self.outputEncodeFunc = self.outputEncoding.encode
        self.outputDecodeFunc = self.outputEncoding.decode
        labels_arr_list = []
        inputs = []; weights = []
        for i in range(Xinput.shape[0]):
            curState = Xinput[i,:,:]
            curStateExog = Xexog[i,:]
            nextState = Xoutput[i,:,:]
            labels = []
            for j in range(curState.shape[1]):
                label = self.outputEncodeFunc(nextState[:,j])
                labels.append(label)
            labels_arr_list.append(labels)
            inputVec = self.ndimTsEmbedding(curState)
            inputVec = np.append(inputVec, curStateExog)
            inputs.append(inputVec)
            weights.append(sig[i])
        inputs = np.array(inputs)
        weights = np.array(weights)

        labels_arr_tensor = np.array(labels_arr_list)

        for i in range(len(labels_arr_list[0][0])):
            labels_arr = labels_arr_tensor[:,:,i]
            biases = np.mean(labels_arr, axis=0)
            labels_arr -= biases
            mod = LeastSquares(params={'lambd': 0.01}).fit(inputs, labels_arr, weights=weights)

            self.mods.append(mod)
            self.biases.append(biases)
        return self
    
    def predict(self, X, Xexog):
        inputvec = self.ndimTsEmbedding(X)
        inputvec = np.append(inputvec, Xexog)
        labels = []
        for i in range(len(self.mods)):
            mod = self.mods[i]
            bias = self.biases[i]
            label = mod.predict(inputvec.reshape(1,-1))[0,:] + bias
            labels.append(label)
        Xnew = []
        for i in range(X.shape[1]):
            label = np.array(list(map(lambda x: x[i], labels)))
            n = self.outputDecodeFunc(label, self.horizon)
            Xnew.append(n)
        return np.array(Xnew).T