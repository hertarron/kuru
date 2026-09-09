import { describe, it, expect } from 'vitest'
import { isCalibrationCancelled } from '../useModelCalibration'

describe('isCalibrationCancelled', () => {
  it('recognises the backend cancellation object', () => {
    expect(
      isCalibrationCancelled({
        code: 'INVALID_ARGUMENT',
        message: 'Invalid configuration argument provided.',
        details: 'Calibration cancelled by user.',
      })
    ).toBe(true)
  })

  it('recognises plain strings and Errors', () => {
    expect(isCalibrationCancelled('Calibration cancelled by user.')).toBe(true)
    expect(isCalibrationCancelled(new Error('Calibration cancelled by user.'))).toBe(
      true
    )
  })

  it('recovers the object from an Error message with a JSON payload', () => {
    const err = new Error(
      'Something failed {"code":"INVALID_ARGUMENT","details":"Calibration cancelled by user."}'
    )
    expect(isCalibrationCancelled(err)).toBe(true)
  })

  it('rejects real failures', () => {
    expect(
      isCalibrationCancelled({
        code: 'OUT_OF_MEMORY',
        message: 'Out of memory.',
      })
    ).toBe(false)
    expect(isCalibrationCancelled(new Error('No backend is configured'))).toBe(false)
    expect(isCalibrationCancelled(undefined)).toBe(false)
    expect(isCalibrationCancelled(null)).toBe(false)
  })
})
