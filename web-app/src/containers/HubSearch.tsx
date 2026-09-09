import { IconSearch, IconX } from '@tabler/icons-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { HUB_COLUMN } from '@/constants/layout'

/**
 * The Hub's search field, on the row under the tabs and inside the same
 * content column as the cards it filters, so it lines up with them.
 */
export function HubSearch({
  value,
  onChange,
  placeholder,
  leading,
  onClear,
  trailing,
  className,
}: {
  value: string
  onChange: (value: string) => void
  placeholder: string
  /** Replaces the magnifier, e.g. with a spinner while a search is running. */
  leading?: React.ReactNode
  /** Called instead of just emptying the field, for pages that clear filters too. */
  onClear?: () => void
  /** Page controls sitting on the search row, to the right of the field. */
  trailing?: React.ReactNode
  className?: string
}) {
  const clear = onClear ?? (() => onChange(''))
  return (
    <div className={cn(HUB_COLUMN, 'flex items-center gap-2', className)}>
      <div className="flex-1 min-w-0 flex items-center gap-2 h-9 px-3 rounded-lg border bg-secondary/20 focus-within:border-primary/50 transition-colors">
        {leading ?? (
          <IconSearch className="shrink-0 text-muted-foreground" size={14} />
        )}
        <input
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-transparent text-sm focus:outline-none"
        />
        {value && (
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Clear search"
            className="shrink-0 -mr-1"
            onClick={clear}
          >
            <IconX size={14} />
          </Button>
        )}
      </div>
      {trailing}
    </div>
  )
}

export default HubSearch
