from enum import Enum
from typing import Dict, Tuple

from imod.mf6.package import Package


class ExchangeType(Enum):
    GWFGWF = "GWF6-GWF6"
    GWFGWT = "GWF6-GWT6"


class ExchangeBase(Package):
    _keyword_map: Dict[str, str] = {}

    @property
    def model_name_1(self):
        return self.dataset["model_name_1"].values[()].take(0)

    @property
    def model_name_2(self):
        return self.dataset["model_name_2"].values[()].take(0)

    def packagename(self) -> str:
        return f"{self.dataset['model_name_1'].values[()]}_{self.dataset['model_name_2'].values[()]}"

    def _filename(self) -> str:
        return f"{self.packagename()}.{self._pkg_id}"

    def get_specification(self) -> Tuple[str, str, str, str]:
        """
        Returns a tuple containing the exchange type, the exchange file name, and the model names. This can be used
        to write the exchange information in the simulation .nam input file
        """
        return (
            self._exchange_type.value,
            self._filename(),
            self.model_name_1,
            self.model_name_2,
        )
