/**
 * Test: redactTraceInput — the fix for "record_trace passes args/output/
 * error straight to Supabase with no redaction" (app/api/mcp/route.ts).
 *
 * Uses Node's built-in node:test runner + tsx (to import .ts directly).
 * Imports the REAL redactTraceInput, same pattern as
 * test-resolve-trace-identity.mjs — no inlined copy to go stale.
 *
 * Run:  npm test   (which runs: node --import tsx --test scripts/test-*.mjs)
 *
 * What this guards against:
 *   - Regressing back to passing raw args/output/error into the
 *     upsert_trace_with_metrics RPC unredacted.
 *   - MCP clients (Claude Desktop, Cursor, Hermes, curl) being the one
 *     ingestion path where PII lands in Supabase, when /api/ingest and
 *     /api/events already redact at the boundary.
 */
import { test, describe } from 'node:test'
import assert from 'node:assert/strict'

import { redactTraceInput } from '../lib/redact-trace-input.ts'

describe('redactTraceInput', () => {
  test('redacts an email in args', () => {
    const result = redactTraceInput({ args: 'contact user@example.com for details' })
    assert.equal(result.args, 'contact [REDACTED] for details')
  })

  test('redacts an API key in output', () => {
    const key = 'sk-' + 'A'.repeat(30)
    const result = redactTraceInput({ output: `used key ${key} to call the model` })
    assert.ok(!result.output.includes(key), 'raw key must not survive redaction')
    assert.match(result.output, /\[REDACTED\]/)
  })

  test('redacts a JWT in error', () => {
    const jwt = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U'
    const result = redactTraceInput({ error: `auth failed: ${jwt}` })
    assert.ok(!result.error.includes(jwt), 'raw JWT must not survive redaction')
  })

  test('redacts a Luhn-valid credit card number but leaves a 16-digit trace id alone', () => {
    const card = '4111 1111 1111 1111' // Luhn-valid Visa test number
    const traceId = '1234567890123456' // 16 digits, not Luhn-valid
    const result = redactTraceInput({
      args: `card ${card}`,
      output: `trace ${traceId}`,
    })
    assert.ok(!result.args.includes('4111'), 'Luhn-valid card must be redacted')
    assert.equal(result.output, `trace ${traceId}`, 'non-Luhn digit strings must pass through')
  })

  test('missing args/output default to empty string, not undefined', () => {
    const result = redactTraceInput({})
    assert.equal(result.args, '')
    assert.equal(result.output, '')
  })

  test('missing error defaults to null, not empty string', () => {
    const result = redactTraceInput({})
    assert.equal(result.error, null)
  })

  test('clean text with no PII passes through unchanged', () => {
    const result = redactTraceInput({
      args: 'model=gpt-4o-mini',
      output: 'tokens=10in/5out',
    })
    assert.equal(result.args, 'model=gpt-4o-mini')
    assert.equal(result.output, 'tokens=10in/5out')
  })
})
