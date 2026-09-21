-- Seed data for the operations store the tool-grounded example queries.
--
-- This is a data source, not a result set: the example's tools run real SQL against it and
-- what comes back depends on the query the agent generated. Point the example at your own
-- database instead and the capture works unchanged.

CREATE TABLE ops_records (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,   -- metric | change | runbook
    service     TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

INSERT INTO ops_records VALUES
 ('metric:queue-depth', 'metric', 'notifications', 'Notification queue depth',
  'notification_queue depth rose from 20 to 18042 messages between 09:06 and 09:48.', '2026-09-18T09:48'),
 ('metric:worker-acks', 'metric', 'email-worker', 'Email worker ack rate',
  'email_worker consumer ack rate fell from 47/s to 0/s at 09:08 and has stayed at zero since.', '2026-09-18T09:48'),
 ('metric:worker-restarts', 'metric', 'email-worker', 'Email worker restart count',
  'email_worker processes restarted 14 times since 09:05, each exiting during startup configuration load.',
  '2026-09-18T09:48'),
 ('metric:order-api-latency', 'metric', 'orders', 'Order API latency',
  'order_api p99 latency is 118ms, inside its normal band of 90-140ms.', '2026-09-18T09:48'),
 ('metric:payment-success', 'metric', 'payments', 'Payment authorization success rate',
  'payment authorization success rate is 99.4 percent, unchanged week over week.', '2026-09-18T09:48'),
 ('metric:cluster-cpu', 'metric', 'platform', 'Cluster CPU utilisation',
  'cluster CPU utilisation is 22 percent, unchanged over the last 24 hours.', '2026-09-18T09:48'),
 ('metric:checkout-conversion', 'metric', 'checkout', 'Checkout conversion rate',
  'checkout conversion rate is 3.1 percent, within the normal weekday range.', '2026-09-18T09:48'),

 ('change:2411', 'change', 'email-worker', 'Release 4.12.0 SMTP credential source',
  'Release 4.12.0 moved email_worker SMTP credentials to the new secrets store, deployed yesterday 17:40.',
  '2026-09-17T17:40'),
 ('change:2409', 'change', 'platform', 'Release 4.12.0 dashboard restyle',
  'Release 4.12.0 restyled the internal operations dashboard with no backend changes.', '2026-09-17T17:40'),
 ('change:2402', 'change', 'payments', 'Payment gateway certificate rotation',
  'Payment gateway TLS certificate rotated three weeks ago, unrelated to release 4.12.0.', '2026-08-28T11:00'),
 ('change:2398', 'change', 'notifications', 'Notification queue retention increase',
  'Notification queue message retention raised from 6 hours to 72 hours last month.', '2026-08-14T09:30'),

 ('runbook-17', 'runbook', 'email-worker', 'Email worker configuration rollback',
  'Revert the email_worker configuration to the previous release and restart workers one at a time, verifying the consumer ack rate recovers before proceeding.', '2026-05-02T00:00'),
 ('runbook-42', 'runbook', 'notifications', 'Notification queue drain',
  'Drain the notification queue with the replay tool once the consumer is healthy again; draining before the consumer recovers loses messages.', '2026-05-02T00:00'),
 ('runbook-23', 'runbook', 'email-worker', 'SMTP credential rotation',
  'Rotate SMTP credentials in the secrets store and trigger a rolling restart so workers reload them.',
  '2026-06-11T00:00'),
 ('runbook-08', 'runbook', 'payments', 'Payment gateway failover',
  'Fail over to the secondary payment gateway when authorization latency exceeds 2 seconds.',
  '2026-04-19T00:00'),
 ('runbook-51', 'runbook', 'checkout', 'Checkout read-only mode',
  'Put checkout into read-only mode during database maintenance windows.', '2026-03-08T00:00');
