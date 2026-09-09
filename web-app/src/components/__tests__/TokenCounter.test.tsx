import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { TokenCounter } from '../TokenCounter'
import { useTokensCount } from '@/hooks/useTokensCount'
vi.mock('@/hooks/useTokensCount', () => ({
  useTokensCount: vi.fn(),
}))

// Mock tooltip components to render inline (Radix Portal + closed state prevents content from appearing in jsdom)
vi.mock('@/components/ui/tooltip', async () => {
  const React = await import('react')
  return {
    TooltipProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    TooltipTrigger: React.forwardRef(({ children, asChild, ...props }: any, ref: any) => {
      if (asChild && React.isValidElement(children)) {
        return React.cloneElement(children as React.ReactElement, { ...props, ref })
      }
      return <span {...props} ref={ref}>{children}</span>
    }),
    TooltipContent: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="tooltip-content">{children}</div>
    ),
  }
})

const mockUseTokensCount = vi.mocked(useTokensCount)

function mockTokens(overrides: Partial<ReturnType<typeof useTokensCount>> = {}) {
  const defaults = {
    tokenCount: 0,
    maxTokens: 1000,
    calculateTokens: vi.fn(),
    ...overrides,
  }
  mockUseTokensCount.mockReturnValue(defaults)
  return defaults
}

describe('TokenCounter', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockTokens()
  })

  it('renders 0.0% when no messages and zero tokens', () => {
    render(<TokenCounter />)
    expect(screen.getAllByText('0.0%').length).toBeGreaterThanOrEqual(1)
  })

  it('renders correct percentage based on token count / max tokens', () => {
    mockTokens({ tokenCount: 500, maxTokens: 1000 })
    render(<TokenCounter />)
    expect(screen.getAllByText('50.0%').length).toBeGreaterThanOrEqual(1)
  })

  it('renders percentage with additionalTokens included', () => {
    mockTokens({ tokenCount: 200, maxTokens: 1000 })
    render(<TokenCounter additionalTokens={300} />)
    expect(screen.getAllByText('50.0%').length).toBeGreaterThanOrEqual(1)
  })

  it('applies destructive styling when over limit (>100%)', () => {
    mockTokens({ tokenCount: 1500, maxTokens: 1000 })
    render(<TokenCounter />)
    const percentElements = screen.getAllByText('150.0%')
    expect(percentElements.length).toBeGreaterThanOrEqual(1)
    const span = percentElements[0]
    expect(span.className).toContain('text-destructive')
  })

  it('applies primary styling when under limit', () => {
    mockTokens({ tokenCount: 500, maxTokens: 1000 })
    render(<TokenCounter />)
    const percentElements = screen.getAllByText('50.0%')
    const span = percentElements[0]
    expect(span.className).toContain('text-foreground')
    expect(span.className).not.toContain('text-destructive')
    expect(span.className).not.toContain('text-amber-500')
  })

  it('calls onClick when clicked', async () => {
    const user = userEvent.setup()
    mockTokens({ tokenCount: 500, maxTokens: 1000 })
    const onClick = vi.fn()
    const { container } = render(<TokenCounter onClick={onClick} />)
    const clickable = container.querySelector('.cursor-pointer')!
    await user.click(clickable)
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('is not interactive without onClick', () => {
    mockTokens({ tokenCount: 500, maxTokens: 1000 })
    const { container } = render(<TokenCounter />)
    expect(container.querySelector('.cursor-pointer')).toBeNull()
  })

  it('renders the SVG progress ring', () => {
    mockTokens({ tokenCount: 500, maxTokens: 1000 })
    const { container } = render(<TokenCounter />)
    const svg = container.querySelector('svg')
    expect(svg).toBeTruthy()
    const circles = container.querySelectorAll('circle')
    expect(circles.length).toBe(2)
  })

  // The badge doubles as the context-visualizer trigger, so it stays mounted
  // before any turn has reported numbers rather than vanishing from the chatbox.
  it('renders a zero badge when neither a limit nor a count is known', () => {
    mockTokens({ tokenCount: 0, maxTokens: undefined })
    render(<TokenCounter />)
    expect(screen.getAllByText('0').length).toBeGreaterThanOrEqual(1)
  })

  it('renders a count-only badge (no percentage) when maxTokens is unavailable but tokens exist', () => {
    mockTokens({
      tokenCount: 1400,
      maxTokens: undefined,
      inputTokens: 1000,
      outputTokens: 400,
      modelDisplayName: 'GPT X',
    })
    render(<TokenCounter />)
    expect(screen.queryByText(/%/)).toBeNull()
    expect(screen.getAllByText('1.4K').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('GPT X')).toBeTruthy()
    expect(screen.getByText('1,400')).toBeTruthy()
  })

  describe('formatNumber helper (via rendered output)', () => {
    it('formats thousands as K', () => {
      mockTokens({ tokenCount: 1000, maxTokens: 2000 })
      const { container } = render(<TokenCounter />)
      expect(container.textContent).toContain('1.0K')
    })

    it('formats millions as M', () => {
      mockTokens({ tokenCount: 1000000, maxTokens: 2000000 })
      const { container } = render(<TokenCounter />)
      expect(container.textContent).toContain('1.0M')
    })

    it('shows raw number below 1000', () => {
      mockTokens({ tokenCount: 500, maxTokens: 1000 })
      const { container } = render(<TokenCounter />)
      expect(container.textContent).toContain('500')
    })
  })

  it('shows token breakdown with Used and Remaining labels', () => {
    mockTokens({ tokenCount: 500, maxTokens: 1000 })
    render(<TokenCounter />)
    const tooltipContent = screen.getByTestId('tooltip-content')
    expect(tooltipContent.textContent).toContain('Used')
    expect(tooltipContent.textContent).toContain('Remaining')
  })

  it('shows Context window header', () => {
    mockTokens({ tokenCount: 500, maxTokens: 1000 })
    render(<TokenCounter />)
    expect(
      screen.getByTestId('tooltip-content').textContent
    ).toContain('Context window')
  })

  it('shows correct remaining tokens', () => {
    mockTokens({ tokenCount: 300, maxTokens: 1000 })
    const { container } = render(<TokenCounter />)
    const tooltipContent = screen.getByTestId('tooltip-content')
    expect(tooltipContent.textContent).toContain('700')
  })

  it('shows 0 remaining when over limit', () => {
    mockTokens({ tokenCount: 1500, maxTokens: 1000 })
    const { container } = render(<TokenCounter />)
    const tooltipContent = screen.getByTestId('tooltip-content')
    expect(tooltipContent.textContent).toContain('Remaining')
    expect(tooltipContent.textContent).toMatch(/Remaining\s*0/)
  })

  it('shows the overflow note and failing-request numbers when isOverflow', () => {
    mockTokens({ tokenCount: 1200, maxTokens: 1000, isOverflow: true })
    const { container } = render(<TokenCounter />)
    const tooltipContent = screen.getByTestId('tooltip-content')
    expect(tooltipContent.textContent).toMatch(/overflow/i)
    expect(screen.getAllByText('120.0%').length).toBeGreaterThanOrEqual(1)
  })

  it('does not show the overflow note when not overflowing', () => {
    mockTokens({ tokenCount: 500, maxTokens: 1000, isOverflow: false })
    render(<TokenCounter />)
    expect(screen.getByTestId('tooltip-content').textContent).not.toMatch(
      /overflow/i
    )
  })

  it('accepts className prop', () => {
    mockTokens({ tokenCount: 0, maxTokens: 1000 })
    const { container } = render(<TokenCounter className="custom-class" />)
    const wrapper = container.querySelector('.custom-class')
    expect(wrapper).toBeTruthy()
  })
})
