import { cn } from '@/lib/utils'

/**
 * The row above every Hub grid: what you are looking at, the filter picker for
 * it, the filters currently applied, and how many results there are. Shared so
 * the three tabs keep the same shape.
 */
export function HubTitleRow({
  title,
  picker,
  chips,
  count,
  className,
}: {
  title: string
  /** The filter/tag picker button. */
  picker?: React.ReactNode
  /** The applied-filter chip row; grows into whatever space is left. */
  chips?: React.ReactNode
  /** Result count, right-aligned. */
  count?: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex items-center gap-2 min-w-0', className)}>
      <h2 className="text-sm font-medium font-studio shrink-0 truncate">
        {title}
      </h2>
      {picker}
      <div className="flex-1 min-w-0">{chips}</div>
      {count != null && (
        <span className="text-xs text-muted-foreground shrink-0">{count}</span>
      )}
    </div>
  )
}

export default HubTitleRow
