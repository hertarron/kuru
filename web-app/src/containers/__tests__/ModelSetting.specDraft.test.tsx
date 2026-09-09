import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'

const hoisted = vi.hoisted(() => ({
  getMtpInfo: vi.fn(),
  updateMtpSettings: vi.fn(),
}))

vi.mock('@/hooks/useServiceHub', () => {
  const hub = {
    models: () => ({
      getMtpInfo: hoisted.getMtpInfo,
      updateMtpSettings: hoisted.updateMtpSettings,
    }),
  }
  return { useServiceHub: () => hub, getServiceHub: () => hub }
})

import { SpecDraftPanel } from '../ModelSetting'

// kuru installs whichever llama.cpp build the user picked, so the panel gates
// on its build number. MTP landed in b9193.
const provider = {
  settings: [
    { key: 'llamacpp_version', controller_props: { value: 'b9500' } },
  ],
} as unknown as ProviderObject

describe('SpecDraftPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    hoisted.updateMtpSettings.mockResolvedValue(undefined)
  })

  it('offers the toggle for a model with MTP heads', async () => {
    hoisted.getMtpInfo.mockResolvedValue({ mtp_layers: 2, mtp: false })
    render(<SpecDraftPanel modelId="glm" provider={provider} />)

    const toggle = await screen.findByRole('switch')
    expect(toggle).toBeEnabled()
  })

  it('reports a model whose MTP is already on as on', async () => {
    hoisted.getMtpInfo.mockResolvedValue({ mtp_layers: 2, mtp: true })
    render(<SpecDraftPanel modelId="glm" provider={provider} />)

    const toggle = await screen.findByRole('switch')
    expect(toggle).toBeChecked()
  })

  it('renders nothing for a model with no MTP heads', async () => {
    hoisted.getMtpInfo.mockResolvedValue({ mtp_layers: 0, mtp: false })
    render(<SpecDraftPanel modelId="llama" provider={provider} />)

    await waitFor(() => expect(hoisted.getMtpInfo).toHaveBeenCalled())
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
  })
})
