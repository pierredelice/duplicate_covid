from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from pandas import DataFrame

street_stopword = [
    'A', 'Y', 'UN', 'UNO', 'AV.', 'AV', 'AVE', 'ESC.', 'ESC .', 'AVENIDA',
    'AMP', 'AMPLIACION', 'XX/XX', 'OO/OO', 'CALLEJON', 'CERRADA', 'CDA.',
    'LA', 'EL', 'ELLA', 'DE', 'DEL', 'PRIVADA', 'PRIV', 'PROLONGACION',
    'PROL.', 'CONOCIDO', 'CONOCIDA', 'S/N', 'S/N ENTRE', 'ENTRE Y ', 'ENTRE',
    ' SIN NOMBRE', 'SIN NOMBRE', 'SIN NOMBRE ', 'No.', 'NO CONOCIDA', 'No .'
]


class Tweak:
    """
    Class documentation
    """

    def __init__(self,
                 data: DataFrame) -> None:
        self.data = data
        self._obj2category()
        self._get_address()
        self._get_calle()
        self._get_num()
        self._get_telephone()
        self._get_colonia()

    def _int2uint32(self) -> None:
        """
        convert integer to uint32
        """
        # type: ignore
        cols = self.data.select_dtypes('int')
        # type: ignore
        self.data = self.data.astype({
            col: "uint16"
            for col in cols
        })

    def _obj2category(self) -> None:
        """
        convert object to category
        """
        # type: ignore
        cols = self.data.select_dtypes('object')
        self.data = self.data.astype({
            col: "category"
            for col in cols
        })

    def _get_address(self) -> None:
        address_data = self.data["DOMICILIO"]
        data = address_data.astype(str)
        data = data.str.strip()
        data = data.str.split(
            'COLONIA: ',
            expand=False
        )
        data = data.str[0]
        data = data.apply(word_tokenize)
        data = data.apply(
            lambda x:
            [word
             for word in x
             if word not in street_stopword]
        )
        data = data.apply(TreebankWordDetokenizer().detokenize)
        data = data.str.replace(
            "\ C.P.+",
            "",
            regex=True
        )
        data = data.str.replace(
            "SIN NOMBRE",
            "",
            regex=True
        )
        data = data.str.replace(
            "No . SIN NUMERO",
            "",
            regex=True
        )
        data = data.str.replace("No . ", "", regex=True)
        self.data["address"] = data

    def _get_calle(self) -> None:
        '''
        calle
        '''
        address_data = self.data["DOMICILIO"]
        # type: ignore
        data = address_data.astype(str)
        data = data.str.strip()
        data = data.str.split(
            'No. ',
            expand=False
        )
        data = data.str[0]
        data = data.str.findall('[A-Z ]+')
        data = data.str.join("")
        data = data.fillna("")
        data = data.apply(word_tokenize)
        data = data.apply(
            lambda x:
            [
                word
                for word in x
                if word not in street_stopword
            ]
        )
        data = data.apply(TreebankWordDetokenizer().detokenize)
        data = data.str.replace(
            "SIN NOMBRE",
            "",
            regex=True
        )
        data = data.str.replace(
            "No . SIN NUMERO",
            "",
            regex=True
        )
        self.data["calle"] = data

    def _get_num(self):
        """
        numero
        """
        address_data = self.data["DOMICILIO"]
        data = address_data.str.split(
            'COLONIA: ',
            expand=False)
        data = data.str[0]
        data = data.str.split(
            'No.',
            expand=False,
        )
        data = data.str[1]
        data = data.str.replace(
            r"\b(.)\1+\b",
            " "
        )
        data = data.str.extract(
            '(\d+)',
            expand=False
        )
        self.data["num"] = data

    def _get_cp(self) -> None:
        """
        Codigo postal
        """
        address_data = self.data["DOMICILIO"]
        data = address_data.astype(str)
        data = data.str.extract(r'(\d{5})')
        data = data.fillna(0)
        data = data.astype('float16')
        self.data["cp"] = data

    def _get_telephone(self) -> None:
        """
        Telefono
        """
        address_data = self.data["DOMICILIO"]
        # type: ignore
        data = address_data.str.extract(r'(\d{10})')
        self.data["telefono"] = data

    def _get_colonia(self):
        """
        colonia
        """
        address_data = self.data["DOMICILIO"]
        # type: ignore
        data = address_data.astype(str)
        data = data.str.strip()
        data = data.str.split(
            'COLONIA: ',
            expand=False
        )
        data = data.str[1]
        data = data.str.split(
            'C.P.',
            expand=False,
        )
        data = data.str[0]
        data = data.str.replace(
            "SIN NOMBRE",
            "",
            regex=True
        )
        data = data.str.replace(
            "No . SIN NUMERO",
            "",
            regex=True
        )
        data = data.str.replace(
            "No . ",
            "",
            regex=True
        )
        self.data["colonia"] = data

    def get_data(self) -> DataFrame:
        """
        Documentation
        """
        return self.data.copy()
