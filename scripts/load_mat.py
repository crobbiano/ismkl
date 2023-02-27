from scipy.io import loadmat
import numpy as np

def load_mat_data(mat_path, shift=True):
    data = loadmat(mat_path)
    train_contacts = {'features': data['dictSetSmall'], 'labels': np.squeeze(data['dictClassSmall']) - shift}
    test_contacts = {'features': data['testSetSmall'], 'labels': np.squeeze(data['testClassSmall']) - shift}
    gen_contacts = {'features': data['validSetSmall'], 'labels': np.squeeze(data['validClassSmall']) - shift}

    return (train_contacts, test_contacts, gen_contacts)