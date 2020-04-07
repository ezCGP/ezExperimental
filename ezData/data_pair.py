"""For Symbolic Regression"""
class DataPair():

    def __init__(self, x, y):
        """
        Class is simple example of a ezData object. Can be used for any problem (like SymbolicRegression) where
        we have only a single training and testing object

        x: raw training data. Shape N x ... ! Any shape is valid as long as it has examples first
        y: labels for the training data. Shape N x ...
        """
        self.x = x
        self.y = y

    def get_data(self):
        """
        Returns: data in a tuple
        """
        return (self.x, self.y)

