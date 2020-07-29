"""
    Classe responsavel por fazer o mapeamento dos arquivos .csv para teste e para treino da rede
    tratando as respostas esperadas para o problema e gerando dicionarios para melhor manipulá-los

    Functions:
        init(self): Inicia a leitura do arquivo.
        handle_input: Funcao que le o arquivo .csv retornando um dicioncario de dados para ser usado na rede neural.
        arquivo(self): Funcao que guarda o objeto do retorno da funcao 'handle_input' na variavel _arquivo.
        arquivo(self,value): Funcao que 'seta' os valores do objeto guardado pela funcao 'arquivo(self)' na variavel _arquivo.
        get_target(self, target): Funcao que retorna o valor esperado dentro da rede de acordo com o target alfanumerico

"""

import csv
from src.env import ARQUIVOS_PARA_TREINO, ARQUIVOS_PARA_TESTE


class Mapper:
    def __init__(self):
        self._arquivos = self.get_multiple_files()
        self.arquivos_teste = self.get_test_file()

    @property
    def arquivos(self):
        return self._arquivos

    @arquivos.setter
    def arquivos(self, value):
        self._arquivos = value

    def get_multiple_files(self):
        result = []
        for arquivo in ARQUIVOS_PARA_TREINO:
            result.append(self.handle_input(arquivo))
        return result

    def get_test_file(self):
        result = []
        for arquivo in ARQUIVOS_PARA_TESTE:
            result.append(self.handle_input(arquivo))
        return result

    def handle_input(self, filename):
        inputs = []
        caminho_arquivo = '../inputs/Part-1/' + filename
        with open(caminho_arquivo, 'rt', encoding="utf-8-sig") as data:
            dados_arquivo = csv.reader(data)

            for linha in dados_arquivo:
                target = self.get_target(linha[-1])
                sample = linha[:-1]
                inputs.append({
                    'target_description': linha[-1],
                    'target': target,
                    'sample': sample
                })

            result = {'nome_problema': filename[:-4],
                      'inputs': inputs}
        return result

    def get_target(self, target):
        dict = {
            'A': [1, 0, 0, 0, 0, 0, 0],
            'B': [0, 1, 0, 0, 0, 0, 0],
            'C': [0, 0, 1, 0, 0, 0, 0],
            'D': [0, 0, 0, 1, 0, 0, 0],
            'E': [0, 0, 0, 0, 1, 0, 0],
            'J': [0, 0, 0, 0, 0, 1, 0],
            'K': [0, 0, 0, 0, 0, 0, 1],
            '0': [0],
            '1': [1]
        }

        """
             O target dos outros problemas e transformado em lista para que a rede seja
            generica para todos os problemas em questao
        """

        return dict[target]