class MetaMod:
    def __init__(self, msw_model, mf6_model):
        self.msw_model = msw_model
        self.mf6_model = mf6_model

    def write(self):
        self.msw_model.write()
        self.msw_model.write()
        self.write_exchanges()

    def write_exchanges(self):
        # TODO
        pass
