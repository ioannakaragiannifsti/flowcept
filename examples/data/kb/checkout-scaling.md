# Scaling checkout for seasonal peaks

Capacity planning for Black Friday and similar traffic peaks. Provision checkout replicas
to three times the weekday baseline, pre-warm caches the evening before, and raise the
autoscaler ceiling for the order API.

This guidance concerns sustained traffic growth over hours, not sudden queue backlogs
caused by a stalled consumer.
