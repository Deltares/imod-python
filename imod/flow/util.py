class Vividict(dict):
    """
    Vividict is used to generate tree structures
    Source: https://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries/19829714#19829714
    """

    def __missing__(self, key):
        value = self[key] = type(self)()  # retain local pointer to value
        return value  # faster to return than dict lookup
