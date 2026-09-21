# Consumer stops acking after malformed credentials

When a queue consumer loads invalid credentials at startup it fails closed: the process
connects to the broker, receives messages, and never acknowledges them. The broker
redelivers, the consumer fails again, and queue depth grows without any error surfacing on
the producer side. Producers stay healthy throughout, which is why the upstream API looks
fine while the backlog climbs.

The signature is an ack rate that drops to zero at a specific timestamp, combined with
repeated consumer restarts that exit during configuration load.
