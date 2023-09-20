class Exchange:
    pass

class GWFGWF(Exchange):

    def __init__(self, model_id1, model_id2, cell_id1, cell_id2, layer):
        self._filename = ""
        self._model_name1 = model_id1
        self._model_name2 = model_id2
        self._cell_id1 = cell_id1
        self._cell_id2 = cell_id2
        self._layer = layer 
