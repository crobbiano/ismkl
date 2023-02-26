from scipy.io import loadmat
import numpy as np

def load_mat_data(mat_path):
    data = loadmat(mat_path)
    train_contacts = {'features': data['dictSetSmall'], 'labels': np.squeeze(data['dictClassSmall'])}
    test_contacts = {'features': data['testSetSmall'], 'labels': np.squeeze(data['testClassSmall'])}
    gen_contacts = {'features': data['validSetSmall'], 'labels': np.squeeze(data['validClassSmall'])}

    return (train_contacts, test_contacts, gen_contacts)