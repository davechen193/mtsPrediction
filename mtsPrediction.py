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

import pywt
class tsEncodingWavelet:
    def __init(self):
        self.coeffShape = None
        pass

    def encode(self, ts):
        coeffs = pywt.wavedec(ts, 'db2', level=8)
        self.coeffShape = [coeff.shape[0] for coeff in coeffs]
        coeffsArr = np.concatenate(coeffs)
        return coeffsArr
    
    def decode(self, encoded, decodeLen):
        coeffs = np.split(encoded, np.cumsum(self.coeffShape)[:-1])
        return pywt.waverec(coeffs, 'db2')

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
        inputs = []; inputsExog = []; weights = []
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
            inputs.append(inputVec)
            inputsExog.append(curStateExog)
            weights.append(sig[i])
        inputs = np.array(inputs)
        inputsExog = np.array(inputsExog)
        weights = np.array(weights)

        labels_arr_tensor = np.array(labels_arr_list)

        for i in range(len(labels_arr_list[0][0])):
            labels_arr = labels_arr_tensor[:,:,i]
            biases = np.mean(labels_arr, axis=0)
            labels_arr -= biases
            inputsStacked = np.column_stack([inputs, inputsExog])
            # model = LeastSquares(params={'lambd': 0.01}).fit(inputsStacked, labels_arr, weights=weights)
            # model = multiOutputRegressor(regressionL1Norm()).fit(inputs, labels_arr)
            from keras.layers import Dense

            # Define the model
            from keras.layers import Input, Concatenate, Dense
            from keras.models import Model

            input1 = Input(shape=(inputs.shape[1],))
            input2 = Input(shape=(inputsExog.shape[1],))
            h1 = Dense(100, activation='relu')(input1)
            h2 = Dense(100, activation='relu')(input2)
            h = Concatenate()([h1, h2])
            h = Dense(32, activation='relu')(h)
            h = Dense(32, activation='relu')(h)
            output = Dense(labels_arr.shape[1])(h)
            model = Model(inputs=[input1, input2], outputs=[output])

            # Compile the model
            model.compile(loss='mean_squared_error', optimizer='adam')

            # Fit the model
            model.fit([inputs, inputsExog], labels_arr, sample_weight=weights, epochs=10, verbose=1)

            # Append the model and biases to the lists
            self.mods.append(model)
            self.biases.append(biases)
            
        return self
    
    def predict(self, X, Xexog):
        inputvecs = []; inputExogvecs = []
        for i in range(X.shape[0]):
            inputvec = self.ndimTsEmbedding(X[i,:,:])
            inputExogvec = Xexog[i,:]
            inputvecs.append(inputvec)
            inputExogvecs.append(inputExogvec)
        inputvecs = np.array(inputvecs)
        inputExogvecs = np.array(inputExogvecs)
        # inputvecs = np.column_stack([inputvecs, inputExogvecs])
        labels = []
        for i in range(len(self.mods)):
            mod = self.mods[i]
            bias = self.biases[i]
            # label = mod.predict(inputvecs) + bias
            label = mod.predict([inputvecs, inputExogvecs]) + bias
            labels.append(label)
        labels = np.array(labels)

        predsAll = []
        for i in range(labels.shape[1]):
            preds = []
            for j in range(labels.shape[2]):
                label = labels[:,i,j]
                n = self.outputDecodeFunc(label, self.horizon)
                preds.append(n)
            predsAll.append(preds)
        predsAll = np.array(predsAll)
        predsAll = predsAll.swapaxes(1, 2)
        return predsAll