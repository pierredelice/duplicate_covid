from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.util import align_tokens
from nltk.tokenize.api import TokenizerI
from nltk.tokenize import word_tokenize
from nltk.tokenize.destructive import MacIntyreContractions
from nltk.tokenize import NLTKWordTokenizer

street_stopword = [
    'A','Y','UN','UNO','AV.','AV','AVE','ESC.','ESC .','AVENIDA','AMP',
    'AMPLIACION','XX/XX','OO/OO','CALLEJON','CERRADA','CDA.','LA','EL',
    'ELLA','DE','DEL','PRIVADA','PRIV','PROLONGACION','PROL.','CONOCIDO',
    'CONOCIDA','S/N','S/N ENTRE','ENTRE Y ','ENTRE',' SIN NOMBRE','SIN NOMBRE',
    'SIN NOMBRE ','No.','NO CONOCIDA','No .'] 


class tweak:
    def __init__(self, convint, convcat) -> None:
        self.convint = convint
        self.convcat = convcat
        ''' convert integer to uint32'''
    def convint(self):
        cols = self.select_dtypes('int')  # type: ignore
        return (self.astype({col:'uint16' for col in cols }))  # type: ignore
        ''' convert object to category'''
    def convcat(self):
        x = self.select_dtypes('object')  # type: ignore
        return(self.astype({col:'category' for col in x }))  # type: ignore

    def address(self):
        cols = (self.astype(str)  # type: ignore
        .str.strip().str.split('COLONIA: ', expand=False).str[0]
        .apply(word_tokenize)
        .apply(lambda x: [word for word in x if not word in street_stopword])
        .apply(TreebankWordDetokenizer().detokenize)
        .str.replace("\ C.P.+", "", regex=True)
        .str.replace("SIN NOMBRE", "", regex=True)
        .str.replace("No . SIN NUMERO", "", regex=True)
        .str.replace("No . ", "", regex=True))
        return cols

    def calle(self):
        '''calle'''       
        colc = (self.astype(str).str.strip()  # type: ignore
        .str.split('No. ', expand=False).str[0]
        .str.findall('[A-Z ]+')
        .str.join("")
        .fillna("")
        .apply(word_tokenize)
        .apply(lambda x: [word for word in x if not word in street_stopword])
        .apply(TreebankWordDetokenizer().detokenize)
        .str.replace("SIN NOMBRE", "", regex=True)
        .str.replace("No . SIN NUMERO", "", regex=True))
        return colc

    def num(self):
        ''''numero'''
        colnum = (self.str.split('COLONIA: ', expand=False, regex=True).str[0]
        .str.split('No.',expand=False,regex=True).str[1]
        .str.replace(r"\b(.)\1+\b", " ")
        .str.extract('(\d+)', expand = False))
        return colnum

    def cp(self):
        '''Codigo postal'''
        colcp = (self.astype(str)
        .str.extract(r'(\d{5})')
        .fillna(0).astype('float16'))
        return colcp

    def telefono(self):
        '''Telefono'''
        coltelefono = (self.str.extract(r'(\d{10})'))  # type: ignore
        return coltelefono

    def colonia(self):
        '''colonia'''
        colcolonia = (self.astype(str).str.strip()  # type: ignore
        .str.split('COLONIA: ', expand=False).str[1]
        .str.split('C.P.', expand=False,regex=True).str[0]
        .str.replace("SIN NOMBRE", "", regex=True)
        .str.replace("No . SIN NUMERO", "", regex=True)
        .str.replace("No . ", "", regex=True))
        return colcolonia
                      