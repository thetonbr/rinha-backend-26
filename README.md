# Rinha de Backend 2026 — detecção de fraude em Zig

Detector de fraude por busca vetorial usando `io_uring`, SIMD AVX2 e índice
IVF quantizado em int16. Atende ao contrato oficial da Rinha 2026 dentro
do orçamento de 1 CPU / 350 MB, distribuído entre duas instâncias da API
atrás do HAProxy.

## Stack

- Zig 0.13.0, alvo `x86_64-linux-musl`, compilado com `-mcpu=x86_64_v3`
  (baseline AVX2, sem AVX-512).
- `io_uring` com `DEFER_TASKRUN + SINGLE_ISSUER + COOP_TASKRUN` em kernels
  que suportam; cai para o modo padrão de forma transparente em kernels
  mais antigos.
- HAProxy 2.9 round-robin entre duas instâncias single-threaded da API.
- IVF-Flat com `nlist = 2048`, `nprobe = 16`, `K' = 50`. Vetores armazenados
  em int16 (escala 10 000, alinhados em 16 lanes); re-rank em float32.
- O arquivo de índice (`index.bin`) é gerado uma única vez no build da
  imagem e compartilhado entre as duas APIs via
  `mmap MAP_SHARED | MAP_POPULATE`.

## Endpoints

O LB escuta em `:9999`, ambas as APIs ouvem em `:8080` internamente.

- `GET /ready` — `200 OK` assim que o índice está mapeado em memória.
- `POST /fraud-score` — corpo segue o schema oficial da Rinha; responde
  com `{"approved": <bool>, "fraud_score": <0.0|0.2|0.4|0.6|0.8|1.0>}`.

`fraud_score = quantidade_de_fraudes_no_top5 / 5` e
`approved = fraud_score < 0.6`.

## Caminho quente

```
read do kernel       -> parser HTTP        ~  2 us
parse JSON           -> PayloadValues      ~  4 us
build do query       -> [16]i16 padded     ~  1 us
top-NPROBE scan      -> centroides f32     ~  3 us  (16 lanes ymm, kernel unico)
scan da invlist      -> distancias int16   ~ 30 us  (vpmovsxwd + vpmulld + vpaddq)
re-rank K'=50        -> distancias f32     ~  4 us  (16 lanes, i16->f32->dist fundido)
top-5 + bit do label -> fraud_count        ~  0 us
resposta pre-bakeada -> write do kernel    ~  1 us
```

Os números acima são alvos otimistas (1 conexão, cache quente, kernel >= 6.x).
A meta de latência é `p99 <= 1 ms`.

## Como rodar

```
docker compose up --build
```

O build acontece em quatro estágios:

1. Toolchain do Zig 0.13.0 baixada para uma camada cacheável.
2. Compilação do binário `api` e do binário auxiliar `build_index`.
3. Execução do `build_index` contra o `references.json.gz` oficial,
   produzindo `/index/index.bin` (3 000 000 vetores, ~96 MB com
   centroides + invlists + bitset de labels).
4. Imagem final `FROM scratch` levando apenas o `api` e o `index.bin`.

## Layout

```
src/
  main.zig                  setup do socket, dispatch para server.run
  http/
    parser.zig              HTTP/1.1 mínimo: GET, POST e Content-Length
    responses.zig           respostas pré-bakeadas em comptime para os 6
                            valores de fraud_score, /ready e respostas 4xx
    server.zig              loop io_uring, multishot accept, submits batched
  io/
    conn.zig                state machine de conexão, ConnPool, handleRead
    uring.zig               wrapper de init com fallback por feature de kernel
  index/
    distance.zig            kernels SIMD (euclideana ao quadrado): f32 wide,
                            i16 -> i64, conversão i16 -> f32 + distância fundida
    format.zig              header binário (magic RNHA), DIM, escala, alinhamento
    ivf.zig                 busca: top-NPROBE -> scan da invlist -> re-rank
    loader.zig              mmap MAP_SHARED | MAP_POPULATE
    topk.zig                heap K' (i64) e insertion sort top-5 com label
  json/
    fraud_payload.zig       parser JSON específico do schema, sem alocação
  vector/
    builder.zig             quantização 14-dim f32 -> i16 com sentinelas
    mcc.zig                 lookup de risco por MCC com perfect hash
    normalize.zig           constantes de clamp/escala fixadas em comptime
    time.zig                parser ISO-8601 (hora, dia da semana, epoch)
build_index/
  parser.zig                leitor streaming de gzip + JSON em array
  kmeans.zig                k-means++ + iterações de Lloyd
  quantize.zig              quantização em lote f32 -> i16 padded
  writer.zig                emissor do index.bin respeitando o contrato de alinhamento
  recall.zig                placeholder de recall (estendido em benchmarks)
  main.zig                  orquestrador end-to-end
conf/haproxy.cfg            round-robin api1/api2 com healthcheck em /ready
docker-compose.yml          api1 (0.425/160) + api2 (0.425/160) + lb (0.150/30)
Dockerfile                  build em 4 estágios, imagem final scratch
```

## Decisões de projeto

- O `index.bin` é mapeado em memória via `mmap` em cada processo, e o
  page cache do kernel reaproveita as mesmas páginas físicas entre
  `api1` e `api2`. É como dois containers de 160 MB conseguem
  compartilhar um índice de ~96 MB sem dobrar o RSS.
- O acumulador da distância int16 é ampliado para `i64` antes do reduce.
  O quadrado por lane cabe em `i32`, mas a soma sobre 16 lanes pode
  chegar a `5.6e9`, acima de `i32_max`. Em `ReleaseFast` o wrap
  silencioso quebraria a monotonicidade do ranking sob entrada
  adversária.
- Toda entrada que alimenta `@intFromFloat` é clampeada e rejeita
  NaN/Inf antes do cast. Sem isso, `ReleaseFast` produziria
  comportamento indefinido em valores fora de range.
- O parser HTTP limita `Content-Length` ao tamanho do buffer de leitura,
  evitando que um cliente malicioso mantenha a conexão em `NeedMore`
  indefinidamente.
- O bloco de vetores em `index.bin` é alinhado a 64 bytes para o loader
  poder expor o slice como `[]align(64) const i16` (loads ymm AVX2). O
  contrato está formalizado em `fmt.VECTORS_BLOCK_ALIGN` e é usado
  tanto pelo writer quanto pelo loader.

## Testes

Testes unitários da runtime e do pipeline de build:

```
zig test src/test_all.zig
```

Os helpers de `build_index/` rodam standalone (eles não fazem parte do
binário `api`):

```
zig test build_index/parser.zig
zig test build_index/kmeans.zig
zig test build_index/quantize.zig
```

## Orçamento de recursos

Soma dos `deploy.resources.limits` dos três containers:
`0.425 + 0.425 + 0.150 = 1.000 CPU`, `160 + 160 + 30 = 350 MB` —
exatamente o teto da competição. O binário `api` tem ~2.3 MB stripped,
é single threaded, linka musl estaticamente e não usa allocator no
caminho de request.
