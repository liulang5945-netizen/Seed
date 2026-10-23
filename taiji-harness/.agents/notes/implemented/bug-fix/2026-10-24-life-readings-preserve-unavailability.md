# Agent Note: Life readings preserve unavailable facts

Status: implemented

English | [中文](2026-10-24-life-readings-preserve-unavailability.zh.md)

## Problem

A missing observation can look like measured inactivity when defaults turn it into `false` or an empty list. A missing life organ can also hide independently available runtime services.

## Decision

Life consumers distinguish unknown readings from observed idle state and empty rosters. Each panel action depends on its own responding source; model context omits unknown training. The [snapshot reference](../../../../docs/subsystems/life.md) owns the field semantics.

## Alternatives considered

**Fill omissions with idle or empty values.** This allows a failed read to enable an action and presents absence of evidence as an observed fact.

**Disable the entire page for one missing source.** This discards healthy independent readings and prevents their controls from working.

## Consequences

Consumers handle unavailable state explicitly. These readings describe one configured runtime; they cannot establish that external research training has stopped or that other checkpoint directories are empty.
