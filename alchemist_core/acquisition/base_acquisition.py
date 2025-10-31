from abc import ABC, abstractmethod

class BaseAcquisition(ABC):
    def __init__(self, search_space, model=None, random_state=42):
        self.search_space = search_space
        self.model = model
        self.random_state = random_state
    
    @abstractmethod
    def update(self, X, y):
        """Update with training data"""
        pass
    
    @abstractmethod
    def select_next(self, candidate_points=None):
        """Select the next point to evaluate"""
        pass
