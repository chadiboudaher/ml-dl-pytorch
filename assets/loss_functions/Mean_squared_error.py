import numpy as np

def MSELoss(data, predictions):
    """ 
    Definition:
        In regression setting, the most commonly used measure is the Mean Squared Error (MSE) 
        
    Usage:
        - MSE small -> very close to the response.
        - MSE large -> very far from the response.
    """
    N = len(data)
    MSE = (1 / N) * np.sum((data - predictions) ** 2)