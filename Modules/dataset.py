from os.path import join
from .tweak import Tweak
from pandas import (
    DataFrame,
    read_csv,
)


class Dataset:
    """
    Documentation
    """

    def __init__(self,
                 params: dict) -> None:
        self.params = params
        self.data = None
        self.dates_col = [
            'FECHREG',
            'FECINISI'
        ]
        self.usefull_cols = [
            'FECHREG',
            'FECINISI',
            'APEPATER',
            'APEMATER',
            'NOMBRE',
            'SEXO',
            'FECNACI',
            'EDAD',
            'CURP',
            'DOMICILIO',
            'CP',
            'TELEFONO',
            'ENTNACI',
            'ENTRESI',
            'MPIORESI',
            'ESINDIGE',
            'HABLEIND',
            'OCUPACIO',
            'FECDEF',
            'CLASCOVID19',
            'RESDEFIN'
        ]
        self._read()
        self._format()

    def _read(self) -> DataFrame:
        filename = join(
            self.params["path_data"],
            self.params["dataset"],
        )
        data = read_csv(
            filename,
            parse_dates=self.dates_col,
            encoding='latin-1',
            low_memory=False,
        )
        self.data = data[self.usefull_cols]

    def _format(self) -> DataFrame:
        """
        Documentation
        """
        tweak = Tweak(self.data)
        self.data = tweak.get_data()

    def get_data(self) -> DataFrame:
        """
        Documentation
        """
        return self.data.copy()
