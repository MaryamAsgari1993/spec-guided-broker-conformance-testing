# Brokers under test

The ICTSS experiments treat each broker as a black box. Versions match Section 4.1 of the paper.

| Broker | Version | Docker image | Host port |
|--------|---------|--------------|-----------|
| EMQX | 4.4.19 | `emqx/emqx:4.4.19` | 1883 |
| NanoMQ | 0.22.4 | `emqx/nanomq:0.22.4` | 1884 |

The oracle itself was trained on Mosquitto 2.0.22 in the companion QRS repository; that broker is **not** re-deployed here.

## Start

```bash
docker compose -f brokers/docker-compose.yml up -d
docker compose -f brokers/docker-compose.yml ps
```

MQTT v5 must be enabled (the default in both images). Replay clients from [replay/README.md](../replay/README.md) against `127.0.0.1:1883` (EMQX) or `127.0.0.1:1884` (NanoMQ).

## Stop

```bash
docker compose -f brokers/docker-compose.yml down
```
