# -*- coding: utf-8 -*-
"""Infraestrutura sem rotulos para predicao seletiva.

Este modulo transforma *features* observaveis no momento da inferencia em um
escore de risco e, depois, em uma decisao ``aceitar/abster``. Por desenho, ele
nao calcula desfechos e nao recebe o rotulo clinico. A avaliacao contra
``verdade``/``severidade`` deve ficar em outro modulo, depois que a politica
estiver congelada.

Convencoes:

* escore maior significa maior risco e, portanto, maior prioridade de abstencao;
* ``PoliticaCoberturaFixa`` aceita exatamente a cota de menor risco no lote;
* empates sao resolvidos por hash de ``item_id`` e semente, sem depender da
  ordem das linhas;
* ``RiscoNomenclaturaLinear`` ajusta apenas centro/escala das features, nunca
  pesos a partir de rotulos;
* o baseline de confianca deve ser aplicado apenas dentro do proprio braco
  ``inn_conf`` ou ``dcb_conf``.

Importar este modulo nunca le arquivos nem executa analises.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


METHOD_VERSION = 'selective-prediction-v0.1'

# Impede que uma especificacao acidental transforme o desfecho em feature.
# A lista e deliberadamente explicita para nao bloquear nomes legitimos como
# ``risco_mapeamento``.
COLUNAS_ALVO_PROIBIDAS = frozenset(
    {
        'verdade', 'severidade', 'rotulo', 'label', 'target', 'y',
        'erro', 'erro_grave', 'subestimacao', 'correto', 'acerto',
    }
)


def _hash_uniforme(valor: object, semente: int) -> float:
    """Numero pseudoaleatorio estavel em [0, 1), independente da ordem."""
    bruto = f'{semente}\x1f{valor}'.encode('utf-8')
    inteiro = int.from_bytes(hashlib.sha256(bruto).digest()[:8], 'big')
    return inteiro / 2**64


def _validar_ids(ids: pd.Series) -> pd.Series:
    if ids.isna().any():
        raise ValueError('IDs ausentes nao sao permitidos na selecao.')
    normalizados = ids.astype(str).str.strip()
    if normalizados.eq('').any():
        raise ValueError('IDs vazios nao sao permitidos na selecao.')
    if normalizados.duplicated().any():
        duplicados = normalizados[normalizados.duplicated(keep=False)].unique()[:5]
        raise ValueError(f'IDs duplicados na selecao: {duplicados.tolist()}')
    return normalizados


def _validar_escores(escores: pd.Series) -> pd.Series:
    numericos = pd.to_numeric(escores, errors='raise').astype(float)
    if not np.isfinite(numericos.to_numpy()).all():
        raise ValueError('Escores de risco devem ser finitos.')
    return numericos


def _validar_cobertura(cobertura: float) -> float:
    cobertura = float(cobertura)
    if not math.isfinite(cobertura) or not 0 <= cobertura <= 1:
        raise ValueError('Cobertura deve estar no intervalo [0, 1].')
    return cobertura


def _coluna_numerica(features: pd.DataFrame, coluna: str) -> pd.Series:
    if coluna not in features.columns:
        raise KeyError(f'Feature ausente: {coluna}')
    if coluna.casefold() in COLUNAS_ALVO_PROIBIDAS:
        raise ValueError(f'Coluna de desfecho nao pode ser feature: {coluna}')
    valores = pd.to_numeric(features[coluna], errors='raise').astype(float)
    if not np.isfinite(valores.to_numpy()).all():
        raise ValueError(f'Feature {coluna} contem valor ausente ou nao finito.')
    return valores


class PontuadorRisco(Protocol):
    """Interface minima para baselines e futuros gatilhos aprendidos."""

    nome: str

    def ajustar(self, features: pd.DataFrame) -> 'PontuadorRisco':
        """Ajusta estado permitido usando somente features, sem rotulos."""

    def pontuar(self, features: pd.DataFrame) -> pd.Series:
        """Retorna um escore por linha; maior significa maior risco."""

    def estado(self) -> dict:
        """Retorna configuracao serializavel para manifesto/reproducao."""


class PoliticaSelecao(Protocol):
    """Converte escores de risco em decisoes sem consultar desfechos."""

    nome: str

    def aplicar(self, escores: pd.Series, ids: pd.Series) -> pd.DataFrame:
        """Retorna uma linha de decisao por item."""

    def estado(self) -> dict:
        """Retorna configuracao serializavel para manifesto/reproducao."""


@dataclass
class RiscoConstante:
    """Baseline sem informacao; util para a politica sem abstencao."""

    valor: float = 0.0
    nome: str = 'constante'

    def ajustar(self, features: pd.DataFrame) -> 'RiscoConstante':
        return self

    def pontuar(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(float(self.valor), index=features.index, name='escore_risco')

    def estado(self) -> dict:
        return {'classe': type(self).__name__, 'nome': self.nome, 'valor': self.valor}


@dataclass
class RiscoAleatorio:
    """Baseline aleatorio reproduzivel, estavel por identificador."""

    coluna_id: str = 'pair_id'
    semente: int = 42
    nome: str = 'aleatorio'

    def ajustar(self, features: pd.DataFrame) -> 'RiscoAleatorio':
        self._ids(features)
        return self

    def _ids(self, features: pd.DataFrame) -> pd.Series:
        if self.coluna_id not in features.columns:
            raise KeyError(f'Coluna de ID ausente: {self.coluna_id}')
        return _validar_ids(features[self.coluna_id])

    def pontuar(self, features: pd.DataFrame) -> pd.Series:
        ids = self._ids(features)
        valores = [_hash_uniforme(x, self.semente) for x in ids]
        return pd.Series(valores, index=features.index, name='escore_risco')

    def estado(self) -> dict:
        return {
            'classe': type(self).__name__, 'nome': self.nome,
            'coluna_id': self.coluna_id, 'semente': self.semente,
        }


@dataclass
class RiscoPorConfianca:
    """Baseline que considera baixa confianca como alto risco.

    Deve ser avaliado somente nas predicoes produzidas pelo mesmo prompt que
    solicitou ``coluna_confianca``; nao deve ser comparado a um braco sem esse
    campo, pois pedir confianca muda o esquema de saida.
    """

    coluna_confianca: str = 'confianca'
    minimo: float = 0.0
    maximo: float = 100.0
    nome: str = 'confianca_auto_relatada'

    def __post_init__(self) -> None:
        if not self.maximo > self.minimo:
            raise ValueError('maximo deve ser maior que minimo na confianca.')

    def ajustar(self, features: pd.DataFrame) -> 'RiscoPorConfianca':
        self._confianca(features)
        return self

    def _confianca(self, features: pd.DataFrame) -> pd.Series:
        valores = _coluna_numerica(features, self.coluna_confianca)
        if ((valores < self.minimo) | (valores > self.maximo)).any():
            raise ValueError(
                f'{self.coluna_confianca} deve estar entre '
                f'{self.minimo:g} e {self.maximo:g}.'
            )
        return valores

    def pontuar(self, features: pd.DataFrame) -> pd.Series:
        confianca = self._confianca(features)
        risco = (self.maximo - confianca) / (self.maximo - self.minimo)
        return risco.rename('escore_risco')

    def estado(self) -> dict:
        return {
            'classe': type(self).__name__, 'nome': self.nome,
            'coluna_confianca': self.coluna_confianca,
            'minimo': self.minimo, 'maximo': self.maximo,
        }


@dataclass(frozen=True)
class EspecificacaoFeature:
    """Peso e direcao pre-especificados de uma feature de nomenclatura."""

    coluna: str
    peso: float = 1.0
    maior_e_maior_risco: bool = True

    def __post_init__(self) -> None:
        if self.coluna.casefold() in COLUNAS_ALVO_PROIBIDAS:
            raise ValueError(f'Coluna de desfecho nao pode ser feature: {self.coluna}')
        if not math.isfinite(float(self.peso)):
            raise ValueError(f'Peso nao finito para {self.coluna}.')


@dataclass
class RiscoNomenclaturaLinear:
    """Escore linear pre-especificado, ajustado sem observar rotulos.

    ``ajustar`` aprende somente mediana e intervalo interquartil no conjunto de
    desenvolvimento. Os pesos e as direcoes devem ser congelados antes do teste.
    Se uma feature tiver IQR zero, sua escala e definida como 1 para preservar
    um resultado finito e registrar o fato no estado exportado.
    """

    especificacoes: Sequence[EspecificacaoFeature]
    padronizar_robusto: bool = True
    nome: str = 'risco_nomenclatura_linear'
    _centros: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _escalas: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _ajustado: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.especificacoes = tuple(self.especificacoes)
        if not self.especificacoes:
            raise ValueError('Ao menos uma feature de nomenclatura e necessaria.')
        nomes = [e.coluna for e in self.especificacoes]
        if len(nomes) != len(set(nomes)):
            raise ValueError('Features de nomenclatura duplicadas.')

    def ajustar(self, features: pd.DataFrame) -> 'RiscoNomenclaturaLinear':
        self._centros = {}
        self._escalas = {}
        for especificacao in self.especificacoes:
            valores = _coluna_numerica(features, especificacao.coluna)
            if self.padronizar_robusto:
                centro = float(valores.median())
                escala = float(valores.quantile(0.75) - valores.quantile(0.25))
                if escala == 0:
                    escala = 1.0
            else:
                centro, escala = 0.0, 1.0
            self._centros[especificacao.coluna] = centro
            self._escalas[especificacao.coluna] = escala
        self._ajustado = True
        return self

    def pontuar(self, features: pd.DataFrame) -> pd.Series:
        if not self._ajustado:
            raise RuntimeError('RiscoNomenclaturaLinear deve ser ajustado antes de pontuar.')
        total = pd.Series(0.0, index=features.index)
        for especificacao in self.especificacoes:
            valores = _coluna_numerica(features, especificacao.coluna)
            z = (
                (valores - self._centros[especificacao.coluna])
                / self._escalas[especificacao.coluna]
            )
            direcao = 1.0 if especificacao.maior_e_maior_risco else -1.0
            total = total + float(especificacao.peso) * direcao * z
        return total.rename('escore_risco')

    def estado(self) -> dict:
        if not self._ajustado:
            raise RuntimeError('Nao ha estado ajustado para exportar.')
        return {
            'classe': type(self).__name__, 'nome': self.nome,
            'padronizar_robusto': self.padronizar_robusto,
            'especificacoes': [asdict(x) for x in self.especificacoes],
            'centros': dict(self._centros), 'escalas': dict(self._escalas),
        }


def _tabela_decisoes(
    escores: pd.Series,
    ids: pd.Series,
    aceitos: np.ndarray,
    ordem_risco: np.ndarray,
    *,
    politica: str,
    cobertura_alvo: float | None,
) -> pd.DataFrame:
    observada = float(np.mean(aceitos)) if len(aceitos) else math.nan
    nome_id = ids.name or 'item_id'
    return pd.DataFrame(
        {
            nome_id: ids.to_numpy(),
            'escore_risco': escores.to_numpy(dtype=float),
            'aceito': aceitos.astype(bool),
            'abster': ~aceitos.astype(bool),
            'ordem_risco': ordem_risco.astype(int),
            'cobertura_alvo': cobertura_alvo,
            'cobertura_observada': observada,
            'politica': politica,
            'metodo_versao': METHOD_VERSION,
        },
        index=escores.index,
    )


@dataclass
class PoliticaCoberturaFixa:
    """Aceita a fracao de menor risco, com cota exata no lote.

    Esta e a politica apropriada para curvas risco-cobertura e para a analise
    primaria em cobertura operacional fixa. A quantidade aceita e o inteiro
    mais proximo de ``n * cobertura``.
    """

    cobertura: float
    semente_desempate: int = 20260817
    nome: str = 'cobertura_fixa'

    def __post_init__(self) -> None:
        self.cobertura = _validar_cobertura(self.cobertura)

    def aplicar(self, escores: pd.Series, ids: pd.Series) -> pd.DataFrame:
        escores = _validar_escores(escores)
        ids = _validar_ids(ids).rename(ids.name)
        if not escores.index.equals(ids.index):
            raise ValueError('Escores e IDs devem ter o mesmo indice e ordem.')

        n = len(escores)
        n_aceitos = int(math.floor(n * self.cobertura + 0.5))
        desempate = [_hash_uniforme(x, self.semente_desempate) for x in ids]
        ordem = pd.DataFrame(
            {
                '_posicao': np.arange(n), '_risco': escores.to_numpy(),
                '_desempate': desempate,
            }
        ).sort_values(['_risco', '_desempate', '_posicao'], kind='mergesort')

        aceitos = np.zeros(n, dtype=bool)
        aceitos[ordem['_posicao'].to_numpy()[:n_aceitos]] = True
        ranks = np.empty(n, dtype=int)
        ranks[ordem['_posicao'].to_numpy()] = np.arange(1, n + 1)
        return _tabela_decisoes(
            escores, ids, aceitos, ranks, politica=self.nome,
            cobertura_alvo=self.cobertura,
        )

    def estado(self) -> dict:
        return {
            'classe': type(self).__name__, 'nome': self.nome,
            'cobertura': self.cobertura,
            'semente_desempate': self.semente_desempate,
        }


@dataclass
class PoliticaLimiar:
    """Politica online com limiar congelado no desenvolvimento.

    Ao contrario da cota por lote, sua cobertura no teste pode mudar. Empates no
    limiar sao aceitos e a cobertura observada deve sempre ser reportada.
    """

    limiar: float
    nome: str = 'limiar_congelado'
    cobertura_calibracao: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.limiar)):
            raise ValueError('Limiar deve ser finito.')
        if self.cobertura_calibracao is not None:
            self.cobertura_calibracao = _validar_cobertura(self.cobertura_calibracao)

    @classmethod
    def calibrar(
        cls,
        escores_desenvolvimento: pd.Series,
        cobertura: float,
    ) -> 'PoliticaLimiar':
        """Calibra apenas pela distribuicao de escores, sem usar rotulos."""
        escores = np.sort(_validar_escores(escores_desenvolvimento).to_numpy())
        cobertura = _validar_cobertura(cobertura)
        if not len(escores):
            raise ValueError('Nao e possivel calibrar limiar sem observacoes.')
        if cobertura == 0:
            # Proximo float abaixo do minimo: nenhum item e aceito na calibracao.
            limiar = float(np.nextafter(escores[0], -np.inf))
        else:
            n_aceitos = max(1, int(math.floor(len(escores) * cobertura + 0.5)))
            limiar = float(escores[min(n_aceitos, len(escores)) - 1])
        return cls(limiar=limiar, cobertura_calibracao=cobertura)

    def aplicar(self, escores: pd.Series, ids: pd.Series) -> pd.DataFrame:
        escores = _validar_escores(escores)
        ids = _validar_ids(ids).rename(ids.name)
        if not escores.index.equals(ids.index):
            raise ValueError('Escores e IDs devem ter o mesmo indice e ordem.')
        aceitos = escores.to_numpy() <= self.limiar
        # Rank informativo; nao participa da decisao por limiar.
        ranks = escores.rank(method='first', ascending=True).to_numpy(dtype=int)
        return _tabela_decisoes(
            escores, ids, aceitos, ranks, politica=self.nome,
            cobertura_alvo=self.cobertura_calibracao,
        )

    def estado(self) -> dict:
        return {
            'classe': type(self).__name__, 'nome': self.nome,
            'limiar': self.limiar,
            'cobertura_calibracao': self.cobertura_calibracao,
        }


@dataclass
class PipelinePredicaoSeletiva:
    """Compoe um pontuador feature-only e uma politica de selecao."""

    pontuador: PontuadorRisco
    politica: PoliticaSelecao
    coluna_id: str = 'pair_id'
    _ajustado: bool = field(default=False, init=False, repr=False)

    def ajustar(self, features_desenvolvimento: pd.DataFrame) -> 'PipelinePredicaoSeletiva':
        self._ids(features_desenvolvimento)
        self.pontuador.ajustar(features_desenvolvimento)
        self._ajustado = True
        return self

    def _ids(self, features: pd.DataFrame) -> pd.Series:
        if self.coluna_id not in features.columns:
            raise KeyError(f'Coluna de ID ausente: {self.coluna_id}')
        return _validar_ids(features[self.coluna_id]).rename(self.coluna_id)

    def aplicar(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self._ajustado:
            raise RuntimeError('Pipeline deve ser ajustado antes de aplicar.')
        ids = self._ids(features)
        escores = self.pontuador.pontuar(features)
        if not escores.index.equals(features.index):
            raise ValueError('Pontuador alterou o indice das observacoes.')
        return self.politica.aplicar(escores, ids)

    def ajustar_aplicar(self, features: pd.DataFrame) -> pd.DataFrame:
        """Atalho adequado apenas para baselines sem estado aprendido."""
        return self.ajustar(features).aplicar(features)

    def estado(self) -> dict:
        if not self._ajustado:
            raise RuntimeError('Pipeline ainda nao foi ajustado.')
        return {
            'metodo_versao': METHOD_VERSION,
            'coluna_id': self.coluna_id,
            'pontuador': self.pontuador.estado(),
            'politica': self.politica.estado(),
        }


def criar_baseline(
    nome: str,
    *,
    cobertura: float,
    coluna_id: str = 'pair_id',
    coluna_confianca: str = 'confianca',
    semente: int = 42,
) -> PipelinePredicaoSeletiva:
    """Fabrica os tres baselines pre-especificados do projeto.

    Valores de ``nome``: ``sem_abstencao``, ``aleatorio`` e ``confianca``.
    """
    nome = nome.strip().casefold()
    if nome == 'sem_abstencao':
        pontuador = RiscoConstante()
        politica = PoliticaCoberturaFixa(1.0, semente_desempate=semente)
    elif nome == 'aleatorio':
        pontuador = RiscoAleatorio(coluna_id=coluna_id, semente=semente)
        politica = PoliticaCoberturaFixa(cobertura, semente_desempate=semente)
    elif nome == 'confianca':
        pontuador = RiscoPorConfianca(coluna_confianca=coluna_confianca)
        politica = PoliticaCoberturaFixa(cobertura, semente_desempate=semente)
    else:
        raise ValueError(
            'Baseline desconhecido. Use sem_abstencao, aleatorio ou confianca.'
        )
    return PipelinePredicaoSeletiva(
        pontuador=pontuador, politica=politica, coluna_id=coluna_id
    )
