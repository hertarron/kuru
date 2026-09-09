import { lorebookToCardBook, type Lorebook } from './lorebook'

const fileName = (book: Lorebook): string => {
  const base = book.name.trim().replace(/[^\w\-. ]+/g, '_').slice(0, 60)
  return `${base || 'lorebook'}.json`
}

/**
 * Hand the book to the user as a `.json` file in the card-embedded dialect.
 * Uses a blob download rather than the save dialog: it is the same path the
 * app's other exports take, and it needs no filesystem permission.
 */
export function exportLorebookFile(book: Lorebook): void {
  const blob = new Blob([JSON.stringify(lorebookToCardBook(book), null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName(book)
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}
