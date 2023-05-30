import spacy


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SpacyModelWrapper(metaclass=Singleton):
    def __init__(self):
        self.model = spacy.load("en_core_web_sm")

    def __call__(self, text):
        return self.model(text)
