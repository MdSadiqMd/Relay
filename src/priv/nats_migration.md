# NATS Migration — Replacing Kafka

## Overview

We are migrating from **Apache Kafka** to **NATS JetStream** as our primary event streaming platform for the payments domain. This migration addresses scalability concerns, operational complexity, and cost.

## Motivation

- **Operational overhead**: Kafka requires dedicated Zookeeper/KRaft clusters, schema registry, and specialized ops knowledge
- **Latency**: Kafka's batching model introduces p99 latencies of 50-100ms that are unacceptable for real-time payment processing
- **Cost**: 3-node Kafka cluster + monitoring stack costs ~$4,200/month
- **Scaling**: Kafka partition rebalancing causes consumer lag spikes during scale events

## New Architecture (NATS JetStream)

- **NATS server cluster**: 3-node NATS cluster with JetStream enabled
- **Streams**:
  - `PAYMENTS` — replaces all `payments.*` Kafka topics
  - `REFUNDS` — replaces `refunds.*` topics
- **Consumers**:
  - `payment-processor` → durable pull consumer on `PAYMENTS`
  - `settlement-engine` → durable push consumer on `PAYMENTS.settled`
  - `notification-service` → push consumer with deliver policy "new"

## Migration Plan

1. **Phase 1** (Q1 2025): Deploy NATS alongside Kafka, dual-write
2. **Phase 2** (Q2 2025): Migrate consumers to NATS, Kafka becomes read-only
3. **Phase 3** (Q3 2025): Decommission Kafka

## Benefits

- **Latency**: p99 < 5ms for event delivery
- **Cost**: ~$800/month (81% reduction)
- **Ops simplicity**: single binary, no Zookeeper, built-in monitoring
- **Native request-reply**: simplifies synchronous payment verification flows

## Risks

- NATS JetStream is less mature for very large-scale event sourcing
- Team needs training on NATS operational patterns
- Consumer exactly-once semantics differ from Kafka's transactional model
