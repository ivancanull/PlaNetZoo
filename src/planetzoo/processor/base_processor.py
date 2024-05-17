from abc import ABC, abstractmethod
# An abstract processor class to process data and restore data
class BaseProcessor(ABC):
    def __init__(self, **kwargs):
        pass    

    @abstractmethod
    def process(self, data):
        pass

    @abstractmethod
    def restore(self, data):
        pass
