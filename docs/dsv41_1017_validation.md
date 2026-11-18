# DeepSeek V4.1 Flash 1017 PD cases

This change includes all three YAML cases and the opt-in Service/client parameter
support they require. It does not change defaults for other cases.

## Validation

69 configuration, topology, parameter and generation regression checks passed,
including mapping 75/121, 66/68 and a synthetic second host pair. Three actual
HTTP timeout integration checks passed in the benchmark container.

## Results (2026-11-17, 66P/68D)

| Case | Successful requests | Measurement |
| --- | ---: | ---: |
| GPQA Diamond, thinking enabled | 198/198 | 92.42424242424242% (183 correct) |
| GSM8K i16384/o1024 C1 | 4/4 | 141.0992 token/s |
| GSM8K i16384/o1024 C64 | 256/256 | 35.1892 token/s |

xLLM: ee1ef466758472934042e6e1b1cf00c520828004.
Service: 2082c47e9160191b5da7f84f589922d109b11fd4.
Base xTask: c89a638 plus the optional parameter adapter included here.
Performance is the AISBench CSV average OutputTokenThroughput, the mean of
per-request output_tokens/E2EL, not aggregate throughput. The C64 final run used
the original streaming completions API; median input was 16384 and every output
had 1024 tokens. Failed or chat-interface diagnostic runs are excluded.

The YAML checks require zero failed requests, the complete request count,
and throughput >= 90% of the measured baseline; improvements do not fail an
upper-bound check. GPQA uses standard DSV4 answer extraction, with no custom
think stripping. Client timeout is 3600 seconds per question, Service default
and maximum deadlines are 3600000 ms, and the whole benchmark limit is 14400
seconds. Service credit limits are 16384 in each case. Queue wait remains the
Service default; no 120-second override is included.

## Target environment

The committed host defaults match existing DSV4 cases (75P/121D). Before running:

1. Set `&prefill_host` and `&decode_host` in each YAML to the target host addresses.
   All topology references reuse those anchors. `HCCL_IF_IP` derives from the
   generated host-local `LOCAL_IP`; do not hardcode it in startup_environment.
2. Prepare 16 compatible NPU ranks on each host, CANN 9.1 / Python 3.12 runtime,
   and the Service/xLLM build artifacts. Compile/deploy compatible artifacts;
   YAML cannot create weights or make incompatible binaries work.
3. Set the model and draft-model directory to the same model revision on both
   hosts (default `/export/home/models/DeepSeek-V4.1-Flash-1017-w8a8/`). Resources
   follow DSV4 at `/export/home/models/xllm_test_060`. Configure different paths
   in each case and the host's existing dependency-package configuration.
4. Provision GPQA Diamond and GSM8K datasets. Configure private SSH credentials
   in `config/ssh_hosts.json`, using the provided example. Never commit credentials.
5. Verify `bond0` matches the actual HCCL/GLOO/TP network interface and that ports
   are free; adjust the case startup_environment and deployment ports if needed.

Run each case with `python main.py --case_name <case_name>`. The three names
are the YAML basenames under `tests/e2e/xllm/deepseek/deepseek_v41_1017/`.
Model, ports, network interface, SSH credentials and device availability remain
explicit environment prerequisites. Only 66/68 was device-tested; the 75/121
configuration and a second mapped topology have CPU-only generation checks.

## Failure evidence

With main ignoring the case's Service budgets, C64 had 107 HTTP 503 failures out
of 256, and GPQA had 19 HTTP 504 and three HTTP 503 failures. Restoring the
explicit case budgets produced the complete successful runs above. This does
not prove the exact cause of each individual 503; no old failures are erased.

Service's full upstream build hit an unrelated test executable link error for
`resolve_model_family`; the production `xllm_master_serving` target built
successfully and runtime dependencies were checked. Service log archive
permissions produced separate collection warnings; they are not counted as
request successes. Full cross-host CPU generation tests are separate from the
recorded device measurements and cannot replace them.

## Ports aligned with the DSV4 reference

| Setting | Port |
| --- | ---: |
| HCCL base | 43432 |
| Progress connection | 8773 |
| etcd client / peer | 22389 / 22380 |
| Model transfer base | 26000 |
| Coordinator HTTP / RPC | 28888 / 28889 |
| Router HTTP / RPC | 38888 / 38889 |
| P/D registration | 7777 |
| Engine local listener | 18004 |

Changing host anchors or ports does not reserve resources. Run sequentially or
choose non-overlapping ports when another DSV4 case is active on the same hosts.
The final anchor/HCCL portability and port changes were CPU-validated after the
recorded overnight device runs; those device results used the earlier ports.
