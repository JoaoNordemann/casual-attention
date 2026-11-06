# Causal Self-Attention

Implementação em PyTorch com dois caminhos: atenção causal manual e
`torch.nn.functional.scaled_dot_product_attention` (SDPA), selecionados por
`use_flash=False` e `use_flash=True`, respectivamente.

## Instalação e testes

Na raiz do projeto, com Python e um ambiente virtual ativo:

```bash
python -m pip install torch pytest numpy
python -m pytest -q
```

O [GitHub Actions](.github/workflows/tests.yml) executa os testes em CPU com
Python 3.12 a cada push e pull request.

## Benchmark: manual vs SDPA

Medição local em CPU AMD64 Family 25 Model 33 Stepping 2 (AuthenticAMD), Windows 11,
Python 3.14.7 e PyTorch 2.15.0+cpu. Configuração: float32, batch=2,
sequência=256, d_model=128, 8 heads, 4 threads, dropout=0, modo treino.
Cada passo inclui forward, loss quadrática média e backward, com gradientes
limpos antes do forward. Foram feitos 10 passos de aquecimento e 50 cronometrados.

| Implementação | Tokens/s | Pico de memória do processo (MiB) |
| --- | ---: | ---: |
| Manual | 102544 | 217.3 |
| SDPA | 130541 | 206.8 |

Nesta execução, SDPA processou aproximadamente 27% mais tokens/s. Os valores
variam com hardware, carga do sistema, versão do PyTorch e formato da entrada.

Para reproduzir a configuração medida, execute na raiz:

```bash
python -m benchmarks.bench
```

Cada implementação roda em um processo separado com a mesma seed. Tokens/s é
`batch × sequência × passos / tempo total`. Em CPU, a memória é o pico de RSS
do processo inteiro, incluindo Python, PyTorch, modelo, entrada e aquecimento;
ela não representa apenas os tensores de atenção nem memória incremental.

Para medir uma GPU CUDA disponível:

```bash
python -m benchmarks.bench --device cuda --batch 8 --seq-len 1024 --d-model 512 --heads 8
```

Em CUDA, o benchmark sincroniza a GPU antes e depois da região cronometrada e
reporta o pico de memória alocada pelo PyTorch após zerar as estatísticas no
fim do aquecimento. Essa métrica inclui os tensores persistentes, exclui memória
reservada sem uso pelo alocador e não é diretamente comparável ao RSS de CPU.
O dtype permanece float32. Use `--help` para ajustar os parâmetros.

A implementação manual materializa scores e probabilidades com formato
`(B, H, T, T)`, cujo custo de memória cresce quadraticamente com a sequência.
SDPA seleciona o backend conforme dispositivo, dtype e compatibilidade;
`use_flash=True` não garante o uso de FlashAttention. A medição acima é em CPU
e não demonstra o desempenho de kernels CUDA.
