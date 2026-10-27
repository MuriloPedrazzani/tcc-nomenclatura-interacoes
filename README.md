# Nomenclatura e predição seletiva de interações medicamentosas

Código e testes do TCC de Ciência da Computação da Pontifícia Universidade Católica do Paraná (PUCPR). Esta entrega reúne a implementação necessária à inspeção das três avaliações do estudo, com testes executáveis sem dados empíricos, credenciais ou chamadas a modelos externos.

**Autores:** Crystofer Samuel Demetino dos Santos; Murilo Chandelier Pedrazzani; Ricardo Ryu Magalhães Makino; Ricardo Vinicius Moreira Vianna.

## Pergunta e escopo

Em um benchmark retrospectivo e exploratório baseado na RENAME e na referência DDInter, como a nomenclatura dos medicamentos se relaciona com a subestimação da categoria de interação, como uma política seletiva pré-especificada se compara à abstenção aleatória de mesma cota e como seus limiares se comportam entre famílias de modelos?

As três frentes são:

1. **Nomenclatura:** comparar DCB e denominação internacional operacional, antes da abstenção, por snapshot, com denominador grave comum aos braços válidos. O contraste é DCB menos denominação internacional.
2. **Política seletiva:** comparar uma política baseada em duas características dos nomes com a expectativa analítica da abstenção aleatória de mesma cota, nas quatro coberturas pré-especificadas: 60%, 70%, 80% e 90%.
3. **Transferência:** aplicar os limiares da família GPT-4.1 à GPT-5.4, sem recalibração ou imposição de cota no alvo, distinguindo cobertura operacional de benefício observado.

“Grave” designa uma categoria da referência DDInter. A denominação internacional é operacional: a classificação estrita como INN não se estende automaticamente a todos os nomes. O estudo descreve um benchmark finito; não estabelece segurança clínica, superioridade populacional ou confirmação prospectiva independente.

## Resultados em imagens

Veja a [galeria de resultados](resultados/README.md), com **quatro gráficos e sete tabelas em PNG** da avaliação congelada: disponibilidade, nomenclatura, política seletiva e transferência entre famílias. As legendas identificam fontes, populações e denominadores. A adição apresenta resultados existentes, sem novas análises.

## Organização

```text
src/
  dados/       # Admissão documental, metadados, características, split e parsing
  analise/     # Preparação, calibração, contratos e três avaliações
tests/         # Testes com exemplos construídos em código
resultados/    # Galeria de gráficos e tabelas dos resultados congelados
requirements.txt
pytest.ini
LICENSE
```

A seleção contém 26 arquivos em `src/` e 15 arquivos de testes. As características de nomenclatura, o tratamento das perdas, a separação das partições e os contratos das avaliações podem ser inspecionados nesses módulos.

| Parte da implementação | Arquivos de entrada para leitura | Verificações incluídas |
|---|---|---|
| Construção e preparação | `src/dados/features_nomenclatura.py`, `src/dados/admissao_metadados_exploratoria.py`, `src/dados/split_admissao_exploratoria.py` | Características dos nomes, vínculos, proveniência, perdas e split |
| Contratos e nomenclatura | `src/analise/contratos_avaliacao_real.py` | Sinal dos contrastes, denominadores comuns, ausências e casos indefinidos |
| Política seletiva | `src/analise/politica_real_exploratoria.py` | Cotas, desempates, baseline analítico e bases operacionais |
| Transferência | `src/analise/transferencia_real_exploratoria.py` | Aplicação inclusiva dos limiares, cobertura e deriva |
| Integração | `src/analise/avaliacao_real_exploratoria.py`, `src/analise/executar_avaliacao_real.py` | Inventário das análises, integridade das entradas e proteção de saídas existentes |

Nomes históricos como `freeze1` e `r6`, assim como referências a documentos em mensagens e constantes, foram preservados no código. Eles não indicam métodos adicionais ativos nesta entrega. Os documentos históricos mencionados não são necessários para executar a suíte selecionada.

## Executar os testes

Recomenda-se **Python 3.12**. As dependências são NumPy, pandas e pytest, nas versões fixadas em `requirements.txt`. A instalação pode exigir acesso à internet; a execução dos testes não requer serviços externos.

No PowerShell, a partir da pasta que contém este README:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider -W error::Warning tests
```

Não é necessário ativar o ambiente virtual. Em Linux ou macOS, com Python 3.12 disponível:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m pytest -q -p no:cacheprovider -W error::Warning tests
```

O `pytest.ini` resolve os caminhos de importação da cópia. Execute os comandos nesta entrega: a suíte do repositório de desenvolvimento completo pode conter outras rotas e dependências.

### Verificação da cópia

Em 26/10/2026, esta cópia foi verificada no Windows com Python **3.12.14**, NumPy **2.1.3**, pandas **2.2.3** e pytest **9.1.1**: **519 testes passaram, com zero avisos, falhas, erros de teste ou testes ignorados**. A execução final levou aproximadamente 29 segundos e tratou qualquer aviso como erro. Os módulos do projeto carregados vieram desta cópia, com plugins externos desativados e acesso à rede e às entradas originais bloqueado.

Esta versão da cópia corrige o tratamento de máscaras booleanas nos módulos de calibração e transferência. A validação estrita é preservada antes da conversão local para o tipo booleano, evitando a inversão obsoleta de booleanos Python e contagens inadequadas de booleanos NumPy em colunas `object`. Foram acrescentados 18 casos de regressão, incluindo imutabilidade das entradas e rejeição de valores inválidos. Seis exemplos construídos também conservaram todos os valores, decisões, limiares e hashes na comparação antes/depois por nomes de colunas.

Os controles de leitura rejeitam componentes de caminho iniciados por ponto dentro da raiz escolhida, sem depender de nomes de ferramentas. Outros 17 casos de regressão cobrem essa proteção, incluindo diretórios de entrada e saída, caminhos absolutos e relativos e artefatos com hash válido. Os bloqueios de caches, bancos, credenciais e escape da raiz são preservados. A regra trata nomes iniciados por ponto, não o atributo de arquivo oculto do Windows.

O código original e os pacotes de resultados congelados foram preservados; esta revisão da cópia não reexecutou nem substitui a implementação vinculada ao congelamento. A compatibilidade com versões futuras das dependências deve ser verificada antes de atualizá-las. Linux e macOS não foram testados nesta entrega.

## O que os testes demonstram

Os testes verificam regras da implementação com exemplos construídos em memória ou em diretórios temporários. Alguns exemplos usam nomes públicos de medicamentos para conferir o processamento textual. Entradas tabulares, bancos de parsing e projetos necessários aos testes são criados pelos próprios testes; eles não substituem os dados empíricos do benchmark.

A suíte verifica, entre outros pontos, a distinção entre posições planejadas, disponíveis e elegíveis; o tratamento de denominador grave igual a zero como indefinido; as cotas iguais da política e do baseline; e a preservação das quatro coberturas. Uma suíte aprovada demonstra o atendimento desses casos de teste, não a validade clínica das previsões.

## Limites de reprodução

Esta entrega permite **inspecionar o código e reproduzir os testes da implementação**. Ela não permite recalcular integralmente os resultados empíricos: respostas históricas, caches, tabelas individuais de predição e referência, pacotes de congelamento e demais entradas reais não acompanham a seleção.

O executor de avaliação é incluído para inspeção e testes com entradas construídas. Ele não é o comando de início rápido desta entrega. A reprodução empírica exige as entradas correspondentes, seus contratos e as condições aplicáveis de acesso e compartilhamento. Não estão incluídas as análises posteriores dos suplementos do artigo.

## Licenças

O arquivo [LICENSE](LICENSE) foi preservado integralmente. Ele atribui MIT ao código de `src/` e `legado/`; esta seleção contém somente `src/`. A presença desse arquivo não estabelece uma licença global nova para todos os testes, documentos ou dados.

As menções a `data/`, `outputs/`, `paper/` e documentos de direitos no LICENSE descrevem o projeto original; essas pastas e documentos não integram esta entrega. O aviso não concede direitos sobre dados de terceiros, que não são redistribuídos aqui. Os avisos e os quatro titulares do código foram mantidos.
