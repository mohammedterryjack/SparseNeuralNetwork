from typing import Tuple, List
from scipy.special import expit

from numpy.random import randn, permutation
from numpy import mean, tile, tanh, ndarray, sum, newaxis, power, arccos, clip, pi, argmax, array
from numpy.linalg import norm
from sklearn.metrics import log_loss

from dfa.utils import ModelSettings

class NeuralNetwork:
    def __init__(self) -> None:
        self.weights1 = randn(ModelSettings.HIDDEN_DIMENSION.value, ModelSettings.INPUT_DIMENSION.value)
        self.weights2 = randn(ModelSettings.OUTPUT_DIMENSION.value, ModelSettings.HIDDEN_DIMENSION.value)
        self.bias1 = randn(ModelSettings.HIDDEN_DIMENSION.value, 1)
        self.bias2 = randn(ModelSettings.OUTPUT_DIMENSION.value, 1)
        self.random_weights = randn(ModelSettings.HIDDEN_DIMENSION.value, ModelSettings.OUTPUT_DIMENSION.value)

    def forward_pass(self, inputs:ndarray) -> ndarray:
        _,batch_size = inputs.shape
        self.activation1 = self._project(
            inputs=inputs,
            weights=self.weights1,
            bias=self.bias1,
            batch_size=batch_size
        )
        self.hidden = self._binary_activation_function(self.activation1)
        self.activation2 = self._project(
            inputs=self.hidden,
            weights=self.weights2,
            bias=self.bias2,
            batch_size=batch_size
        )
        return expit(self.activation2)
    
    def direct_feedback_alignment(self, error:ndarray, inputs:ndarray) -> None:
        self.weights2 += self._dWeights(error, self.hidden)
        self.bias2 += self._dBias(error)

        error_activation = self._dActivation(error)
        self.weights1 += self._dWeights(error_activation, inputs)
        self.bias1 += self._dBias(error_activation)


    def fit(self, train_inputs:ndarray, train_outputs:ndarray) -> Tuple[List[float],List[float],List[float]]:        
        _,dataset_size = train_inputs.shape
        n_batches = dataset_size//ModelSettings.BATCH.value

        loss_curve,training_error,angles = list(),list(),list()
        for epoch in range(ModelSettings.EPOCHS.value):
            train_inputs_shuffled,train_outputs_shuffled = self._shuffle_data(dataset_size, train_inputs,train_outputs)
            loss,error,angle = self._partial_fit(n_batches,train_inputs_shuffled,train_outputs_shuffled)
            loss_curve.append(loss)
            training_error.append(error/dataset_size)
            angles.append(angle)
            print(f'\nLoss at epoch {epoch+1}: {round(loss/dataset_size,2)}.  Training error: {round(training_error[-1],2)}')
        return loss_curve,training_error,angles

    def _partial_fit(self,n_batches:int, train_inputs:ndarray,train_outputs:ndarray) -> Tuple[float,float,float]:
        loss,error,angle = 0.,0.,0.
        for batch in range(n_batches):
            train_inputs_sampled,train_outputs_sampled = self._sample_data(batch,train_inputs,train_outputs)
            l,e,angle = self._fit_batch(train_inputs_sampled, train_outputs_sampled)
            loss += l
            error += e
            print(f"\rbatch: {batch} - {100*round(batch/n_batches,2)}%",end="")
        return loss, error, angle

    def _fit_batch(self, train_inputs:ndarray, train_outputs:ndarray) -> Tuple[float,float,float]:
        predicted_output_logits = self.forward_pass(train_inputs)
        self.direct_feedback_alignment(
            error=predicted_output_logits - train_outputs, 
            inputs=train_inputs
        )
        return (
            self._train_loss(predicted_output_logits,train_outputs),
            self._train_error(predicted_output_logits,train_outputs),
            self._average_angle(predicted_output_logits - train_outputs)
        )

    def _average_angle(self, error:ndarray) -> float:
        c = self._c(error)        
        dHidden =  self._dHidden(error)
        Lk = self._Lk(dHidden, c) 
        return self._angle(Lk, c)
        
    def _dActivation(self,error:ndarray) -> ndarray:
        return (self.random_weights @ error) *  ( 1 - tanh(self.activation1)**2 )

    def _dHidden(self, error:ndarray) -> ndarray:
        return mean( self.random_weights @ error , axis=1)[:, newaxis]

    def _c(self, error:ndarray) -> ndarray:
        activationInverse = expit( self.activation2 ) 
        activationInverseMultInverse = 1 - activationInverse
        a = activationInverse * activationInverseMultInverse
        e = error * a
        return mean( self.weights2.T @ e, axis=1)[:, newaxis]
    
    @staticmethod
    def _dWeights(error:ndarray, inputs:ndarray) -> ndarray:
        dWeights = -error @ inputs.T 
        return ModelSettings.LEARNING_RATE.value * dWeights

    @staticmethod
    def _dBias(error:ndarray) -> ndarray:
        dbias = -sum(error,axis=1)[:,newaxis]
        return ModelSettings.LEARNING_RATE.value * dbias

    @staticmethod
    def _angle(Lk:ndarray, c:ndarray) -> float:
        return arccos(clip(Lk*NeuralNetwork._inverse(c), -1., 1.)) * 180/pi

    @staticmethod
    def _Lk(dHidden:ndarray, c:ndarray) -> ndarray:
        return ((dHidden.T @ c) * NeuralNetwork._inverse(dHidden))[0, 0]

    @staticmethod
    def _project(inputs:ndarray, weights:ndarray, bias:ndarray, batch_size:int) -> ndarray:
        return weights @ inputs + tile(bias, batch_size)
    
    @staticmethod
    def _inverse(x:ndarray) -> ndarray:
        return power(norm(x), -1)

    @staticmethod
    def _shuffle_data(dataset_size:int, train_inputs:ndarray, train_outputs:ndarray) -> Tuple[ndarray,ndarray]:
        shuffled_ids = permutation(dataset_size)
        return train_inputs[:, shuffled_ids], train_outputs[:, shuffled_ids]

    @staticmethod
    def _sample_data(batch_number:int, train_inputs_shuffled:ndarray, train_outputs_shuffled:ndarray) -> Tuple[ndarray,ndarray]:
        return (
            train_inputs_shuffled[:, batch_number*ModelSettings.BATCH.value:(batch_number+1)*ModelSettings.BATCH.value],
            train_outputs_shuffled[:, batch_number*ModelSettings.BATCH.value:(batch_number+1)*ModelSettings.BATCH.value]
        )

    @staticmethod
    def _train_error(predicted:ndarray, expected:ndarray) -> int:
        return sum(argmax(predicted, axis=0) != argmax(expected, axis=0))
    
    @staticmethod
    def _train_loss(predicted:ndarray, expected:ndarray) -> float:
        return log_loss(expected, predicted)

    @staticmethod
    def _binary_activation_function(x:ndarray) -> ndarray:
        return array(x > ModelSettings.BINARISATION_THRESHOLD.value,dtype=int)