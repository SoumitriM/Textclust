from river.feature_extraction import BagOfWords as RiverBagOfWords

class BagOfWords(RiverBagOfWords):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transform_one(self, x, **kwargs):
        x = super().transform_one