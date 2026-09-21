# Draining a notification backlog safely

Never drain a notification queue while its consumer is still unhealthy. Draining replays
messages into a consumer that cannot acknowledge them, which loses the messages without
delivering them.

The safe order is: restore the consumer, confirm the ack rate has recovered, then replay
the backlog in batches while watching for duplicate deliveries. Customers may receive
delayed confirmations, which is preferable to silent loss.
