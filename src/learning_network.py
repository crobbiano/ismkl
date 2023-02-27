import numpy as np
from src.kernel_matrix import KernelMatrix
from src.recursive_omp import RecursiveOMP
import pickle
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
        self.kernels = kernels
        self.residual_norm = residual_norm
        self._training_sample_order = training_sample_order
        self.verbose = verbose

        self.L0 = None
        self.K00 = None
        self.W0 = None

        self.min_sparsity = min_sparsity

    def training_sample_order(self, num_samples):
        if self._training_sample_order is None:
            self._training_sample_order = np.arange(num_samples)
            np.random.shuffle(self._training_sample_order)
        return self._training_sample_order

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
        print(f'{network_type} saved to {network_path}.')


    def load_trained_network(self, network):
        if isinstance(network, BaseNetwork):
            self.L0 = network.L0
            self.K00 = network.K00
            self.W0 = network.W0
        elif isinstance(network, str):
            try:
                with open(network, 'rb') as fh:
                    network = pickle.load(fh)
                    self.L0 = network.L0
                    self.K00 = network.K00
                    self.W0 = network.W0
            except FileNotFoundError:
                raise FileNotFoundError(f'Network path {network} not found.')

    def train(self):
        pass

    def score_samples(self, sample_batch, features):
        K = KernelMatrix(x=sample_batch, y=features, kernels=self.kernels).matrix
        scores = np.dot(K, self.W0)
        return scores

    def classifier(self, samples, labels):
        scores = self.score_samples(samples, self.features)
        num_classes = self.L0.shape[1]
        labels_estimated = np.argmax(scores, axis=1)
        labels_truth = labels
        correct_class = labels_estimated == labels_truth
        return labels_estimated, correct_class




class BaseNetwork(LearningNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, network_path):
        try:
            with open(network_path, 'rb') as fh:
                network = pickle.load(fh)
                self.L0 = network.L0
                self.K00 = network.K00
                self.W0 = network.W0
        except FileNotFoundError:
            raise FileNotFoundError(f'Network path {network_path} not found.')

    def save(self, network_path: str = './base_network.p'):
        super().save(network_path, 'base_network.p')

    def train(self):
        n_values = np.max(self.labels) + 1
        self.L0 = np.eye(n_values)[self.labels]
        self.K00 = KernelMatrix(self.features, self.features, self.kernels).matrix
        self.W0 = RecursiveOMP(self.K00, [], self.L0, residual_norm=self.residual_norm).run()

        base_results = self.classifier(self.features, self.labels)
        print(f'Base accuracy: {np.count_nonzero(base_results[1])/len(base_results[1])}')


class TrainedNetwork(LearningNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, network_path):
        try:
            with open(network_path, 'rb') as fh:
                network = pickle.load(fh)
                self.L0 = network.L0
                self.K00 = network.K00
                self.W0 = network.W0
        except FileNotFoundError:
            raise FileNotFoundError(f'Network path {network_path} not found.')

    def save(self, network_path: str = './trained_network.p'):
        super().save(network_path, 'trained_network.p')

    def learn_batch(self, features, sample_batch, label_batch):
        scores = self.score_samples(sample_batch, features)
        num_classes = self.L0.shape[1]

        labels_estimated = np.argmax(scores, axis=1)
        labels_truth_vec = np.eye(num_classes)[label_batch]
        num_added = 0

        if not np.all(labels_estimated == label_batch):
            # FIXME? This should be looking at misclassified samples instead of confidence in scores
            scores.sort(axis=1)
            sorted_diff = scores[:, -1] - scores[:, -2]
            # This is weighing top labels against second labels
            guessed_wrong = sorted_diff <= 0.2

            wrong_features = sample_batch[:, guessed_wrong]
            wrong_truth = labels_truth_vec[guessed_wrong, :]

            labels_small = np.concatenate([self.labels, label_batch[guessed_wrong]], axis=0)
            l0_small = np.concatenate([self.L0, wrong_truth], axis=0)

            K10 = KernelMatrix(sample_batch, self.features, self.kernels).matrix
            Kmat = np.concatenate([self.K00, K10], axis=0)

            L0 = np.concatenate([self.L0, labels_truth_vec], axis=0)
            labels = np.concatenate([self.labels, label_batch], axis=0)

            # FIXME why oh why copy()
            W = RecursiveOMP(Kmat, self.W0.copy(), L0, self.residual_norm).run()

            num_coefficients_to_delete = np.max(np.sum(W != 0, axis=0) - np.sum(self.W0 != 0, axis=0))
            if num_coefficients_to_delete < self.min_sparsity:
                self.train_features = np.concatenate([self.train_features, sample_batch], axis=1)
                self.K00 = Kmat
                self.W0 = W
                self.L0 = L0
                self.labels = labels
            else:
                K10 = KernelMatrix(wrong_features, self.features, self.kernels).matrix
                K01 = KernelMatrix(self.train_features, wrong_features, self.kernels).matrix
                K11 = KernelMatrix(wrong_features, wrong_features, self.kernels).matrix
                Kmat = np.concatenate([np.concatenate([self.K00, K01], axis=1), np.concatenate([K10, K11], axis=1)],
                                      axis=0)
                W = RecursiveOMP(Kmat, self.W0, l0_small, self.residual_norm).run()
                self.features = np.concatenate([self.features, wrong_features], axis=1)
                self.train_features = np.concatenate([self.train_features, wrong_features], axis=1)
                self.K00 = Kmat
                self.W0 = W
                self.L0 = l0_small
                self.labels = labels_small
                num_added += np.count_nonzero(guessed_wrong)
            return num_added

    def train(self, val_features, val_labels, batch_size: int = 50):
        available = self.training_sample_order(val_features.shape[1])
        atoms_added = 0

        # FIXME
        batch_size = np.min([batch_size, val_features.shape[1]])

        for batch in range(0, val_features.shape[1], batch_size):
            selected_samples = available[batch:batch+batch_size]

            sample_batch = val_features[:, selected_samples]
            label_batch = val_labels[selected_samples]
            new_atoms = self.learn_batch(self.features, sample_batch, label_batch)
            atoms_added += new_atoms
            print(f'Batch {batch//batch_size} -- Added {new_atoms} atoms')

            batch_results = self.classifier(sample_batch, label_batch)
            print(f'Batch accuracy: {np.count_nonzero(batch_results[1])/len(batch_results[1])}')

        print(f'Added {atoms_added}/{val_features.shape[1]} atoms total')

