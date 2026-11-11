"""Learned record metadata inputs without changing raw byte copy targets."""

from __future__ import annotations

import math

from torch import nn


def install_metadata(model) -> None:
    model.record_metadata = nn.Linear(4, model.c.d)
    model.record_provenance = nn.Linear(model.c.d, model.c.d, bias=False)
    nn.init.normal_(model.record_metadata.weight, std=0.1 / math.sqrt(4))
    nn.init.zeros_(model.record_metadata.bias)
    nn.init.normal_(model.record_provenance.weight, std=0.1 / math.sqrt(model.c.d))


def metadata_vector(model, record, world_time):
    if type(record.version) is not int or record.version < 0:
        raise ValueError("Record version must be a nonnegative integer")
    if not math.isfinite(record.observed_at) or record.observed_at > world_time:
        raise ValueError("Invalid record observation time")
    stamp = math.copysign(math.log1p(abs(record.observed_at)), record.observed_at)
    values = model.native_w.new_tensor(
        [
            math.log1p(record.version),
            stamp,
            math.log1p(world_time - record.observed_at),
            float(record.writable),
        ]
    )
    provenance = model.text_units((record.source + "\0" + record.scope).encode()).mean(0)
    return model.record_metadata(values) + model.record_provenance(provenance)
