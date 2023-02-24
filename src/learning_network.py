import numpy as np
from src.kernel_matrix import KernelMatrix
from src.recursive_omp import RecursiveOMP
import pickle
import logging
from typing import List
from schema import Schema

class LearningNetwork:
    def __init__(self,
                 features: np.array = None,
                 labels: np.array = None,
                 kernels: dict = None,
                 residual_norm: float = 0.1,
                 training_sample_order: List[int] = None,
                 min_sparsity: int = 60,
                 verbose: bool = True):
        self.features = features
        self.train_features = features
        self.labels = labels
        # kernels needs to look like a dict:
        #   {'<kernel_name>': {'param1': <value>, 'param2': <value>, etc}
        self.kernels = kernels
        self.residual_norm = residual_norm
        self._training_sample_order = training_sample_order
        self.verbose = verbose

        self.L0 = None
        self.K00 = None
        self.W0 = None

        self._validate_variables()

        self.multi_class_classifier = MultiClassClassifier()
        self.min_sparsity = min_sparsity

    @property
    def training_sample_order(self):
        pass

    def _validate_features(self):
        if self.features is None:
            raise TypeError(f'Features are empty. Need them to train the networks')
        if self.labels is None:
            raise TypeError(f'Labels are empty. Need them to train the networks')
        if self.kernels is None:
            raise TypeError(f'Kernels are empty. Need them to train the networks')

    def load(self, network_path):
        pass

    def save(self, network_path: str = './network.p', network_type: str = 'network'):
        with open(network_path, 'wb') as fh:
            pickle.dump(self)
        logging.info(f'{network_type} saved to {network_path}.')


    def load_trained_network(self, network_path):
        try:
            with open(network_path, 'rb') as fh:
                network = pickle.load(fh)
                self.L0 = network.L0
                self.K00 = network.K00
                self.W0 = network.W0
        except FileNotFoundError:
            logging.error(f'Network path {network_path} not found.')

    def train(self):
        pass

    def score_samples(self, sample_batch, features):
        K = KernelMatrix(X=sample_batch, Y=features, kernels=self.kernels).matrix
        scores = np.dot(K, self.W0)
        return scores




class BaseNetwork(LearningNetwork):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)

    def load(self, network_path):
        try:
            with open(network_path, 'rb') as fh:
                network = pickle.load(fh)
                self.L0 = network.L0
                self.K00 = network.K00
                self.W0 = network.W0
        except FileNotFoundError:
            logging.error(f'Network path {network_path} not found.')

    def save(self, network_path: str = './base_network.p'):
        super().save(network_path, 'base_network.p')

    def train(self):
        n_values = np.max(self.values) + 1
        self.L0 = np.eye(n_values)[self.values]
        self.K00 = KernelMatrix(self.features, self.features, self.kernel_dict).matrix
        self.W0 = RecursiveOMP(self.K00, [], self.L0, residual_norm=self.residual_norm).run()


class TrainedNetwork(LearningNetwork):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)

    def training_sample_order(self, num_samples):
        self._training_sample_order = np.random.random_sample(0, num_samples, num_samples)
        return self._training_sample_order
    def load(self, network_path):
        try:
            with open(network_path, 'rb') as fh:
                network = pickle.load(fh)
                self.L0 = network.L0
                self.K00 = network.K00
                self.W0 = network.W0
        except FileNotFoundError:
            logging.error(f'Network path {network_path} not found.')

    def save(self, network_path: str = './trained_network.p'):
        super().save(network_path, 'trained_network.p')

    def learn_batch(self, features, sample_batch):
        scores = self.score_samples(sample_batch, features)
        num_classes = self.L0.shape[1]

        labels_estimated = np.argmax(scores, axis=1)
        labels_truth_vec = np.eye(num_classes)[self.labels]
        num_added = 0

        if not np.all(labels_estimated == self.labels):
            sorted = scores.sort(axis=1)
            sorted_diff = sorted[:, -1] - sorted[:, -2]
            guessed_wrong = sorted_diff <= 0.2

            wrong_features = sample_batch[:, guessed_wrong]
            wrong_truth = labels_truth_vec[guessed_wrong, :]
            l_small = np.concatenate([self.features, wrong_truth], axis=0)
            K10 = KernelMatrix(sample_batch, self.features, self.kernels)
            Kmat = np.concatenate([self.K00, K10], axis=0)
            L = np.concatenate([self.labels, labels_truth_vec], axis=0)
            W = RecursiveOMP(Kmat, self.W0, L, self.residual_norm).run()

            num_coefficients_to_delete = np.max(np.sum( W!=0, axis=0) - np.sum(self.W0 != 0, axis=0))
            if num_coefficients_to_delete < self.min_sparsity:
                self.train_features = np.concatenate([self.train_features, sample_batch], axis=1)
                self.K00 = Kmat
                self.W0 = W
                self.labels = L
            else:
                K10 = KernelMatrix(sample_batch, self.features, self.kernels)
                K01 = KernelMatrix(self.train_features, sample_batch, self.kernels)
                K11 = KernelMatrix(sample_batch, sample_batch, self.kernels)
                Kmat = np.concatenate(np.concatenate([self.K00, K01], axis=1),
                                      np.concatenate([K10, K11], axis=1), axis=0)
                W = RecursiveOMP(Kmat,self.W0, l_small, self.residual_norm).run()
                self.features = np.concatenate([self.features, sample_batch], axis=1)
                self.train_features = np.concatenate([self.train_features, sample_batch], axis=1)
                self.K00 = Kmat
                self.W0 = W
                self.labels = l_small
            return sample_batch.shape[2]

    def train(self, val_features, batch_size: int = 50):
        available = np.arange(len(val_features))
        atoms_added = 0

        for batch in range(0, len(val_features), batch_size):
            random_samples = self.training_sample_order(available)[batch:batch+batch_size]
            selected_samples = available[random_samples]
            del available[random_samples]

            sample_batch = val_features[selected_samples]
            atoms_added += self.learn_batch(val_features, sample_batch)

