# Optional budgets scoped to one YAML

The current main accepts explicit Coordinator/Router topology but does not pass
Service credit/deadline options to its binaries. This change forwards only the
five supported positive integer budget fields when explicitly present under
`deployment.xllm_service`; absent fields retain the binary defaults. It does
not change global budgets, queue waits, or request deadlines.

AISBench accuracy cases may set `benchmark.acc.request_timeout` (seconds) and
`request_retries` (total attempts). The adapter preserves shared HTTP sessions
and applies the timeout to each actual HTTP request. It does not change GPQA
prompts, answer extraction, or thinking behavior.

Example for a long-generation accuracy case:

```yaml
benchmark:
  acc:
    request_timeout: 3600
    request_retries: 2
deployment:
  xllm_service:
    batch_credit_max_grant_units: 16384
    batch_credit_max_grant_units_per_router_tenant: 16384
    default_request_deadline_ms: 3600000
    max_request_deadline_ms: 3600000
```

`flow_max_queue_wait_ms` is optional and omitted by default. It is independent
of the request deadline. Validation: 64 framework regression checks and three
actual HTTP timeout integration checks passed on 2026-11-17. Full device-run
results are recorded separately; this document does not claim they passed.
