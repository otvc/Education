import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    probas = np.sum(y,axis=0)/y.shape[0]
    entropy_value = - np.sum(probas*np.log(probas + EPS))
    return entropy_value
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    probas = np.sum(y,axis=0)/y.shape[0]
    gini_value = 1 - np.sum(np.power(probas, 2))
    return gini_value
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    return y.var()

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    median = np.median(y)
    mm_value = np.mean(np.abs(y - median))
    return mm_value


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        mask_less = X_subset[:, feature_index] < threshold
        left_mask = np.where(mask_less)
        right_mask = np.where(~mask_less)
        X_left, y_left = X_subset[left_mask], y_subset[left_mask]
        X_right, y_right = X_subset[right_mask], y_subset[right_mask]
        
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        mask_less = X_subset[:, feature_index] < threshold
        left_mask = np.where(mask_less)
        right_mask = np.where(~mask_less)
        y_left, y_right= y_subset[left_mask], y_subset[right_mask]
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        N = X_subset.shape[0]
        max_crit_value = 0
        feature_index, threshold=0,0
        
        for supposed_index in range(X_subset.shape[1]):
            for supposed_threshold in X_subset[:,supposed_index]:
                sub_crit_value = self.criterion(y_subset)
                y_left, y_right = self.make_split_only_y(supposed_index, supposed_threshold, X_subset, y_subset) 
                left_crit_value = self.criterion(y_left)
                right_crit_value = self.criterion(y_right)
                full_crit_value = sub_crit_value-(y_left.shape[0] * left_crit_value + y_right.shape[0] * right_crit_value)/N
#                 print(full_crit_value)
                if max_crit_value < full_crit_value:
                    feature_index = supposed_index
                    threshold = supposed_threshold
                    max_crit_value = full_crit_value
        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        
        #Точки останова
        new_node = Node(-1, -1)
        probas = None
        unique_cls = np.unique(y_subset)
        stop_equation = X_subset.shape[0] < self.min_samples_split or self.max_depth == self.depth or unique_cls.shape[0] == 1
        if self.classification and stop_equation:
            probas = np.mean(y_subset,axis=0)
            new_node.proba = probas
            new_node.value = np.argmax(probas)
            return new_node
        elif stop_equation:#regression
            if self.criterion_name == 'variance':
                value = y_subset.mean()
            else:
                value = np.median(y_subset)
            new_node.value = value
            return new_node
        
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
        
        new_node = Node(feature_index, threshold)
        current_depth = self.depth
        self.depth += 1
        new_node.left_child = self.make_tree(X_left, y_left)
        self.depth =  current_depth + 1
        new_node.right_child = self.make_tree(X_right, y_right)
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
        if self.debug:
            print('tree is maked')
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        pred_count = 0
        y_predicted = np.zeros((X.shape[0], 1)).astype(np.int32)
        current_node = self.root
        stack_nodes = [(current_node, np.full((X.shape), True))]
        l_index = 0
        r_index = 0
        while stack_nodes:
            current_node, current_mask = stack_nodes.pop()
            if current_node.left_child == None:
                y_predicted[current_mask.sum(axis = -1) == current_mask.shape[-1]] = current_node.value
                pred_count += np.sum(current_mask)
                continue
            idx = current_node.feature_index
            mask_less = current_mask.copy()
            mask_more = current_mask.copy()
            expr = X[:, idx] < current_node.value
            mask_less[:, idx] = np.logical_and(mask_less[:, idx], expr)
            mask_more[:, idx] = np.logical_and(mask_more[:, idx], ~expr)
            stack_nodes.append((current_node.right_child, mask_more))
            stack_nodes.append((current_node.left_child, mask_less))
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        pred_count = 0
        y_predicted_probs = np.zeros((X.shape[0], self.n_classes)).astype(np.float)
        current_node = self.root
        stack_nodes = [(current_node, np.full((X.shape), True))]
        l_index = 0
        r_index = 0
        while stack_nodes:
            current_node, current_mask = stack_nodes.pop()
            if current_node.left_child == None:
                y_predicted_probs[current_mask.sum(axis = -1) == current_mask.shape[-1]] = current_node.proba
                continue
            idx = current_node.feature_index
            mask_less = current_mask.copy()
            mask_more = current_mask.copy()
            expr = X[:, idx] < current_node.value
            mask_less[:, idx] = np.logical_and(mask_less[:, idx], expr)
            mask_more[:, idx] = np.logical_and(mask_more[:, idx], ~expr)
            stack_nodes.append((current_node.right_child, mask_more))
            stack_nodes.append((current_node.left_child, mask_less))
        return y_predicted_probs
