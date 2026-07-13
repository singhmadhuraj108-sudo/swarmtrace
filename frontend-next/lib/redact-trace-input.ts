/**
 * Redacts the free-text fields (args/output/error) of an incoming trace
 * before it's persisted.
 *
 * Extracted out of app/api/mcp/route.ts so this can be unit tested directly
 * (see scripts/test-redact-trace-input.mjs), same pattern as
 * resolve-trace-identity.ts — instead of only being exercised through a
 * full MCP tool-call round trip against a live Supabase connection.
 *
 * BEFORE this fix: record_trace passed params.args/output/error straight
 * into the upsert_trace_with_metrics RPC with no redaction at all.
 * /api/ingest (lib/validate-ingest.ts) and /api/events
 * (lib/redact.ts::redactEventData) both redact at the ingest boundary as
 * defense-in-depth — for a client posting directly, bypassing the Python
 * SDK's own redaction (swarmtrace/redact.py). MCP was the one ingestion
 * path that skipped this, even though going through the SDK is optional
 * here — any MCP client (Claude Desktop, Cursor, Hermes, per this route's
 * own docstring) can call record_trace directly with raw args/output/error.
 *
 * AFTER: args/output/error go through the same redact() used by
 * /api/ingest before being sent to Supabase.
 */
import { redact } from './redact'

export interface RedactableTraceInput {
  args?: string
  output?: string
  error?: string
}

export interface RedactedTraceInput {
  args: string
  output: string
  error: string | null
}

export function redactTraceInput(input: RedactableTraceInput): RedactedTraceInput {
  return {
    args: redact(input.args ?? '') ?? '',
    output: redact(input.output ?? '') ?? '',
    error: redact(input.error) ?? null,
  }
}
