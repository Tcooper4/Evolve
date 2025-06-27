class ValidationError(Exception):
    """Raised when model validation fails."""
    pass

class BaseModel:
    def __init__(self): 
        pass 

class ModelRegistry:
    """Stub class to hold registered models. Expand as needed."""
    registry = {}

    @classmethod
    def register(cls, name, model_class=None):
        cls.registry[name] = model_class or f"Placeholder for {name}"

    @classmethod
    def get(cls, name):
        return cls.registry.get(name) 