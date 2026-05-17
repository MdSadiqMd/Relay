# Kafka Event Streaming Architecture

## Overview

Our payments platform uses **Apache Kafka** as the primary event streaming backbone. All inter-service communication in the payments domain flows through Kafka topics.

## Architecture

- **Broker cluster**: 3-node Kafka cluster (v3.4) running on dedicated instances
- **Schema Registry**: Confluent Schema Registry for Avro schema management
- **Topics**:
  - `payments.created` — new payment events
  - `payments.settled` — settlement confirmations
  - `payments.failed` — payment failure events
  - `refunds.requested` — refund initiation events

## Consumer Groups

- `payment-processor` — processes incoming payments
- `settlement-engine` — handles settlement workflows
- `notification-service` — sends payment notifications
- `audit-logger` — persists all events for compliance

## Configuration

```yaml
bootstrap.servers: kafka-1:9092,kafka-2:9092,kafka-3:9092
acks: all
retries: 3
enable.idempotence: true
```

## Operational Notes

- Kafka is the **single source of truth** for event ordering
- All services must be idempotent consumers
- Topic retention: 7 days for operational topics, 90 days for audit topics
- Monitoring via Kafka Manager + Prometheus JMX exporter
