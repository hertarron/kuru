/* eslint-disable @typescript-eslint/no-explicit-any */
import { memo, useState, useCallback, useEffect, useMemo, useRef } from 'react'
import type { UIMessage, ChatStatus } from 'ai'
import { RenderMarkdown } from './RenderMarkdown'
import { cn } from '@/lib/utils'
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '@/components/ui/dialog'
import { ChainOfThoughtGroup } from './message/ChainOfThoughtGroup'
import {
  CHAT_STATUS,
  CONTENT_TYPE,
  type MessagePartLike,
  type PartEntry,
} from './message/types'
import { CopyButton } from './CopyButton'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { formatDate } from '@/utils/formatDate'
import { useModelProvider } from '@/hooks/useModelProvider'
import { useInterfaceSettings } from '@/hooks/useInterfaceSettings'
import { useMessageErrors } from '@/stores/message-errors'
import {
  IconRefresh,
  IconPlayerPlay,
  IconPaperclip,
  IconAlertTriangle,
  IconChevronLeft,
  IconChevronRight,
  IconGitFork,
  IconUserCircle,
} from '@tabler/icons-react'
import { EditMessageDialog } from '@/containers/dialogs/EditMessageDialog'
import { DeleteMessageDialog } from '@/containers/dialogs/DeleteMessageDialog'
import TokenSpeedIndicator from '@/containers/TokenSpeedIndicator'
import { extractFilesFromPrompt, FileMetadata } from '@/lib/fileMetadata'
import { Button } from '@/components/ui/button'
import { PromptProgress } from '@/components/PromptProgress'
import { useServiceHub } from '@/hooks/useServiceHub'
import { useToolApprovalRequests } from '@/hooks/useToolApprovalRequests'
import { parseCitationsFromToolOutput } from '@/lib/citation-parser'
import type { RagCitation, WebCitation } from '@/components/Citations'
import { useGroundingStore } from '@/stores/grounding-store'
import { useWebCitationStore } from '@/stores/web-citation-store'
import { WebSourcesRow } from '@/components/WebSourcesRow'
import { injectCitationMarkers } from '@/lib/grounding'

export type MessageItemProps = {
  message: UIMessage
  isFirstMessage: boolean
  isLastMessage: boolean
  status: ChatStatus
  reasoningContainerRef?: React.RefObject<HTMLDivElement | null>
  isReasoningAtBottom?: boolean
  onReasoningScroll?: () => void
  onReasoningScrollToBottom?: () => void
  onRegenerate?: (messageId: string) => void
  onContinue?: (messageId: string) => void
  onEdit?: (messageId: string, newText: string) => void
  /** Duplicate this assistant reply as a new childless version. */
  onFork?: (messageId: string) => void
  onDelete?: (messageId: string) => void
  versionInfo?: { index: number; count: number }
  onSwitchVersion?: (messageId: string, dir: -1 | 1) => void
  isAnimating?: boolean
  hideActions?: boolean
  /** Character portrait, shown above and left of an assistant row. */
  charAvatarUrl?: string
  /** Persona portrait, shown above and right of a user row. */
  userAvatarUrl?: string
  /** Coding characters: no persona in the scene, no user portrait at all. */
  userAvatarHidden?: boolean
}

export const MessageItem = memo(
  ({
    message,
    isFirstMessage,
    isLastMessage,
    status,
    isAnimating,
    hideActions,
    reasoningContainerRef,
    isReasoningAtBottom,
    onReasoningScroll,
    onReasoningScrollToBottom,
    onRegenerate,
    onContinue,
    onEdit,
    onFork,
    onDelete,
    versionInfo,
    onSwitchVersion,
    charAvatarUrl,
    userAvatarUrl,
    userAvatarHidden,
  }: MessageItemProps) => {
    const { t } = useTranslation()
    const selectedModel = useModelProvider((state) => state.selectedModel)
    const coloredUserBubble = useInterfaceSettings((s) => s.coloredUserBubble)
    const metadata = message.metadata as Record<string, unknown> | undefined
    const messageError = useMessageErrors((s) => s.errors[message.id])
    const createdAt = (metadata?.createdAt as Date) ?? new Date()
    const [previewImage, setPreviewImage] = useState<{
      url: string
      filename?: string
    } | null>(null)
    // Survives the close: the dialog's exit animation needs the <img> still
    // mounted while it zooms out, but previewImage is already null by then.
    const lastPreviewUrlRef = useRef(previewImage?.url)
    if (previewImage) lastPreviewUrlRef.current = previewImage.url
    // One Radix dialog per message would mean hundreds in a long chat, so the
    // zoom popup only enters the tree once this message's image is clicked.
    // It stays after that -- unmounting it would cut the exit animation.
    const [previewMounted, setPreviewMounted] = useState(false)
    const openPreview = useCallback(
      (image: { url: string; filename?: string }) => {
        setPreviewMounted(true)
        setPreviewImage(image)
      },
      []
    )

    const handleRegenerate = useCallback(() => {
      onRegenerate?.(message.id)
    }, [onRegenerate, message.id])

    const handleContinue = useCallback(() => {
      onContinue?.(message.id)
    }, [onContinue, message.id])

    const isStopped = metadata?.stopped === true

    /** Portrait above the message. It sits in the flow and reserves its own
        height, so the size is free to change without retuning the margins
        between messages. The user's runs 20% smaller than the character's. */
    const charAvatarSize = 'size-24 rounded-xl'
    const userAvatarSize = 'size-[69px] rounded-xl'
    // Which portrait this row actually draws. The spacing below reads these,
    // so a row can never reserve height for a portrait it does not render.
    const showCharPortrait = message.role === 'assistant' && !!charAvatarUrl
    const showUserPortrait = message.role === 'user' && !userAvatarHidden
    const avatarImg = (
      url: string | undefined,
      size: string,
      extra?: string
    ) =>
      url ? (
        <img
          src={url}
          alt=""
          className={cn(
            size,
            'object-cover border cursor-zoom-in shrink-0',
            extra
          )}
          onClick={() => openPreview({ url })}
        />
      ) : null
    const userPlaceholder = (extra?: string) => (
      <div
        className={cn(
          userAvatarSize,
          'border bg-secondary/40 flex items-center justify-center shrink-0',
          extra
        )}
      >
        <IconUserCircle size={46} className="text-muted-foreground" />
      </div>
    )

    const handleEdit = useCallback(
      (newText: string) => {
        onEdit?.(message.id, newText)
      },
      [onEdit, message.id]
    )

    const handleDelete = useCallback(() => {
      onDelete?.(message.id)
    }, [onDelete, message.id])

    // Get image URLs from file parts for the edit dialog
    const imageUrls = useMemo(() => {
      return message.parts
        .filter((part) => {
          if (part.type !== 'file') return false
          const filePart = part as { type: 'file'; url?: string; mediaType?: string }
          return filePart.url && filePart.mediaType?.startsWith('image/')
        })
        .map((part) => (part as { url: string }).url)
    }, [message.parts])

    // A tool part is "pending" until it reaches a terminal state. While any
    // tool on the last assistant message is still pending the turn isn't
    // done — the model will resume once the tool result arrives, even if the
    // SDK briefly reports status as 'ready' between the tool-call stream and
    // the follow-up request.
    const hasPendingToolCall = useMemo(() => {
      if (!isLastMessage || message.role !== 'assistant') return false
      return message.parts.some((part) => {
        if (!part.type?.startsWith('tool-')) return false
        const state = (part as { state?: string }).state
        return (
          state !== 'output-available' &&
          state !== 'output-error' &&
          state !== 'output-denied'
        )
      })
    }, [isLastMessage, message.role, message.parts])

    const pendingApprovals = useToolApprovalRequests((s) => s.pending)
    const awaitingApproval = useMemo(() => {
      if (!hasPendingToolCall) return false
      return message.parts.some((part) => {
        const toolCallId = (part as { toolCallId?: string }).toolCallId
        return Boolean(toolCallId && pendingApprovals[toolCallId])
      })
    }, [hasPendingToolCall, message.parts, pendingApprovals])

    const isStreaming =
      (isLastMessage &&
        (status === CHAT_STATUS.STREAMING ||
          status === CHAT_STATUS.SUBMITTED)) ||
      hasPendingToolCall

    // Aggregate RAG citations in part order and record each rag tool part's
    // base offset, so its card numbers/anchors continue the same global
    // sequence the inline superscript markers use.
    const { ragCitations, citationOffsets, webCitations } = useMemo(() => {
      const out: RagCitation[] = []
      const web: WebCitation[] = []
      const offsets = new Map<number, number>()
      if (message.role === 'assistant') {
        const parts = message.parts as any[]
        for (let i = 0; i < parts.length; i++) {
          const part = parts[i]
          if (!part.type?.startsWith('tool-')) continue
          if (part.state !== 'output-available') continue
          const parsed = parseCitationsFromToolOutput(part.output)
          if (parsed?.kind === 'rag') {
            offsets.set(i, out.length)
            out.push(...parsed.citations)
          } else if (parsed?.kind === 'web') {
            web.push(...parsed.citations)
          }
        }
      }
      return { ragCitations: out, citationOffsets: offsets, webCitations: web }
    }, [message.parts, message.role])

    const serviceHub = useServiceHub()
    const grounding = useGroundingStore((s) => s.byMessageId[message.id])
    const ensureGrounding = useGroundingStore((s) => s.ensure)

    const assistantText = useMemo(() => {
      if (message.role !== 'assistant') return ''
      return (message.parts as any[])
        .filter((p) => p.type === CONTENT_TYPE.TEXT && p.text)
        .map((p) => p.text)
        .join('\n')
    }, [message.parts, message.role])

    useEffect(() => {
      if (isStreaming) return
      if (!assistantText || !ragCitations.length) return
      const rag = serviceHub.rag()
      if (!rag.embed) return
      ensureGrounding(
        message.id,
        assistantText,
        ragCitations,
        rag.embed.bind(rag)
      )
    }, [
      isStreaming,
      assistantText,
      ragCitations,
      message.id,
      ensureGrounding,
      serviceHub,
    ])

    const setWebCitations = useWebCitationStore((s) => s.setForMessage)
    useEffect(() => {
      if (!webCitations.length) return
      setWebCitations(message.id, webCitations)
    }, [webCitations, message.id, setWebCitations])

    // Extract file metadata from message text (for user messages with attachments)
    const attachedFiles = useMemo(() => {
      if (message.role !== 'user') return []

      const textParts = message.parts.filter(
        (part): part is { type: 'text'; text: string } =>
          part.type === CONTENT_TYPE.TEXT
      )

      if (textParts.length === 0) return []

      const { files } = extractFilesFromPrompt(textParts[0].text)
      return files
    }, [message.parts, message.role])

    // Get full text content for copy button
    const getFullTextContent = useCallback(() => {
      return message.parts
        .filter(
          (part): part is { type: 'text'; text: string } =>
            part.type === CONTENT_TYPE.TEXT
        )
        .map((part) => part.text)
        .join('\n')
    }, [message.parts])

    const renderTextPart = (
      part: { type: 'text'; text: string },
      partIndex: number
    ) => {
      if (!part.text || part.text.trim() === '') {
        return null
      }

      const isLastPart = partIndex === message.parts.length - 1

      // For user messages, extract and clean the text from file metadata
      const displayText =
        message.role === 'user'
          ? extractFilesFromPrompt(part.text).cleanPrompt
          : part.text

      if (
        !displayText.trim() &&
        message.role === 'user' &&
        attachedFiles.length === 0
      ) {
        return null
      }

      return (
        <div key={`${message.id}-${partIndex}`} className="w-full">
          {message.role === 'user' ? (
            <div className="flex justify-end w-full h-full text-start wrap-break-word whitespace-normal">
              <div
                className={cn(
                  'relative p-2 rounded-md inline-block max-w-[80%]',
                  coloredUserBubble
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary text-foreground'
                )}
              >
                {/* Show attached files if any */}
                {attachedFiles.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-3">
                    {attachedFiles.map((file: FileMetadata, idx: number) => (
                      <div
                        key={`file-${idx}-${file.id}`}
                        className="flex items-center gap-1.5 px-2 py-1 rounded-sm bg-secondary text-secondary-foreground border text-xs"
                      >
                        <IconPaperclip
                          size={14}
                          className="text-muted-foreground"
                        />
                        <span className="font-medium">{file.name}</span>
                        {file.injectionMode && (
                          <span className="text-muted-foreground">
                            ({file.injectionMode})
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {displayText && (
                  <div dir="auto" className="select-text whitespace-pre-wrap">
                    {displayText}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <>
              <RenderMarkdown
                content={
                  grounding && !isStreaming
                    ? injectCitationMarkers(
                        part.text,
                        grounding.sentenceCitations,
                        `cite-${message.id}`
                      )
                    : part.text
                }
                isStreaming={isStreaming && isLastPart}
                messageId={message.id}
                isAnimating={isAnimating}
              />
            </>
          )}
        </div>
      )
    }

    const renderFilePart = (part: MessagePartLike, partIndex: number) => {
      const isImage = part.mediaType?.startsWith('image/')
      const isAudio =
        part.mediaType === 'audio/wav' || part.mediaType === 'audio/mpeg'
      const isVideo = part.mediaType?.startsWith('video/')

      if (isAudio && part.url) {
        const justify =
          message.role === 'user' ? 'justify-end' : 'justify-start'
        return (
          <div
            key={`${message.id}-${partIndex}`}
            className={`flex ${justify} w-full my-2`}
          >
            <audio
              controls
              src={part.url}
              className="max-w-[80%] rounded-md"
            />
          </div>
        )
      }

      if (isVideo && part.url) {
        const justify =
          message.role === 'user' ? 'justify-end' : 'justify-start'
        return (
          <div
            key={`${message.id}-${partIndex}`}
            className={`flex ${justify} w-full my-2`}
          >
            <video
              controls
              src={part.url}
              className="max-w-[80%] max-h-80 rounded-md border"
            />
          </div>
        )
      }

      if (message.role === 'user' && isImage && part.url) {
        return (
          <div
            key={`${message.id}-${partIndex}`}
            className="flex justify-end w-full my-2"
          >
            <div className="flex flex-wrap gap-2 max-w-[80%] justify-end">
              <div className="relative">
                <img
                  src={part.url}
                  alt={part.filename || 'Uploaded attachment'}
                  className="size-20 rounded-lg object-cover border cursor-pointer"
                  onClick={() =>
                    openPreview({ url: part.url!, filename: part.filename })
                  }
                />
              </div>
            </div>
          </div>
        )
      }

      if (message.role === 'assistant' && isImage && part.url) {
        return (
          <div key={`${message.id}-${partIndex}`} className="my-2">
            <img
              src={part.url}
              alt={part.filename || 'Generated image'}
              className="max-w-full rounded-md cursor-pointer"
              onClick={() =>
                openPreview({ url: part.url!, filename: part.filename })
              }
            />
          </div>
        )
      }

      return null
    }

    const renderedParts = useMemo(() => {
      const parts = message.parts as MessagePartLike[]
      const elements: React.ReactNode[] = []
      const isCotPart = (t: string) =>
        t === CONTENT_TYPE.REASONING || t.startsWith('tool-')

      // Walk parts sequentially and flush the reasoning/tool trace whenever a
      // non-empty answer (text/file) interrupts it, so content emitted between
      // two reasoning blocks renders as a normal message.
      let cotEntries: PartEntry[] = []
      let groupSeq = 0
      const flushCot = (hasFollowing: boolean) => {
        if (cotEntries.length === 0) return
        elements.push(
          <ChainOfThoughtGroup
            key={`${message.id}-cot-${groupSeq++}`}
            entries={cotEntries}
            messageId={message.id}
            totalParts={parts.length}
            isStreaming={isStreaming}
            hasFollowingContent={hasFollowing}
            awaitingApproval={awaitingApproval}
            citationOffsets={citationOffsets}
            reasoningContainerRef={reasoningContainerRef}
            isReasoningAtBottom={isReasoningAtBottom}
            onReasoningScroll={onReasoningScroll}
            onReasoningScrollToBottom={onReasoningScrollToBottom}
          />
        )
        cotEntries = []
      }

      for (let i = 0; i < parts.length; i++) {
        const part = parts[i]
        const t = part.type
        if (isCotPart(t)) {
          cotEntries.push({ part, index: i })
          continue
        }
        if (t === CONTENT_TYPE.TEXT) {
          if (!part.text || part.text.trim() === '') continue
          flushCot(true)
          elements.push(
            renderTextPart(part as { type: 'text'; text: string }, i)
          )
          continue
        }
        if (t === CONTENT_TYPE.FILE) {
          flushCot(true)
          elements.push(renderFilePart(part, i))
        }
      }
      flushCot(false)
      return elements
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
      message.parts,
      isStreaming,
      isReasoningAtBottom,
      grounding,
      awaitingApproval,
      citationOffsets,
    ])

    // Version switching rebuilds the rendered path from the store; doing it
    // mid-generation would clobber the in-flight stream, so the control stays
    // visible but inert while the thread is busy.
    const isThreadBusy =
      status === CHAT_STATUS.STREAMING || status === CHAT_STATUS.SUBMITTED
    const versionNav =
      versionInfo &&
      versionInfo.count > 1 &&
      onSwitchVersion ? (
        <div className="flex items-center gap-0.5 text-muted-foreground">
          <button
            type="button"
            className="hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
            disabled={isThreadBusy || versionInfo.index <= 1}
            onClick={() => onSwitchVersion(message.id, -1)}
            title="Previous version"
          >
            <IconChevronLeft size={14} />
          </button>
          <span className="tabular-nums">
            {versionInfo.index}/{versionInfo.count}
          </span>
          <button
            type="button"
            className="hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
            disabled={isThreadBusy || versionInfo.index >= versionInfo.count}
            onClick={() => onSwitchVersion(message.id, 1)}
            title="Next version"
          >
            <IconChevronRight size={14} />
          </button>
        </div>
      ) : null

    return (
      <div
        className={cn(
          'w-full mb-4 group/message relative',
          // Jan's rhythm for a portrait-less chat. A portrait brings its own
          // vertical space, so the two must not stack.
          message.role === 'user' &&
            !isFirstMessage &&
            !showUserPortrait &&
            'mt-8'
        )}
      >
        {/* In the flow, so it reserves its own height. Hanging in the margin
            only worked on the character's side: assistant markdown is full
            width, so the persona portrait landed on its last line. */}
        {showCharPortrait && (
          <div className="mb-1 flex justify-start">
            {avatarImg(charAvatarUrl, charAvatarSize)}
          </div>
        )}
        {showUserPortrait && (
          <div className="mb-2 flex justify-end">
            {userAvatarUrl
              ? avatarImg(userAvatarUrl, userAvatarSize)
              : userPlaceholder()}
          </div>
        )}

        {/* Render message parts */}
        {renderedParts}

        {message.role === 'assistant' && !isStreaming && webCitations.length > 0 && (
          <WebSourcesRow citations={webCitations} />
        )}

        {isLastMessage &&
          message.role === 'assistant' &&
          !awaitingApproval &&
          (hasPendingToolCall || status === CHAT_STATUS.SUBMITTED) && (
            <div className="mt-2">
              <PromptProgress hideIdle={hasPendingToolCall} />
            </div>
          )}

        {typeof messageError === 'string' && messageError.length > 0 && (
          <div className="mt-2 flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm">
            <IconAlertTriangle
              size={16}
              className="mt-0.5 shrink-0 text-destructive"
            />
            <div className="flex-1 min-w-0">
              <div className="font-medium text-destructive">
                Generation failed
              </div>
              <div className="text-muted-foreground break-words">
                {messageError}
              </div>
            </div>
            {selectedModel && onRegenerate && status !== CHAT_STATUS.STREAMING &&
              status !== CHAT_STATUS.SUBMITTED && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRegenerate}
                  className="shrink-0"
                >
                  <IconRefresh size={14} />
                  <span>Regenerate</span>
                </Button>
              )}
          </div>
        )}

        {/* Message actions for user messages */}
        {message.role === 'user' && !hideActions && (
          <div className="flex items-center justify-end gap-1 text-muted-foreground text-xs opacity-0 transition-opacity group-hover/message:opacity-100 focus-within:opacity-100">
            <span className="text-muted-foreground">
              {formatDate(createdAt)}
            </span>
            {versionNav}
            <CopyButton text={getFullTextContent()} />

            {onEdit && status !== CHAT_STATUS.STREAMING &&
              status !== CHAT_STATUS.SUBMITTED && (
              <EditMessageDialog
                message={getFullTextContent()}
                imageUrls={imageUrls.length > 0 ? imageUrls : undefined}
                onSave={handleEdit}
              />
            )}

            {onDelete && status !== CHAT_STATUS.STREAMING &&
              status !== CHAT_STATUS.SUBMITTED && (
              <DeleteMessageDialog
                onDelete={handleDelete}
                deletesReply={message.role === 'user'}
              />
            )}
          </div>
        )}

        {/* Message actions for assistant messages (non-tool) */}
        {message.role === 'assistant' && (
            <div className="flex items-center gap-2 text-muted-foreground text-xs">
              {!isStreaming && (
                <span className="text-muted-foreground">
                  {formatDate(createdAt)}
                </span>
              )}
              <div className="flex items-center gap-1">
                {versionNav}
                <div
                  className={cn(
                    'flex items-center gap-1',
                    (isStreaming || hideActions) && 'hidden'
                  )}
                >
                <CopyButton text={getFullTextContent()} />

                {onEdit && !isStreaming && (
                  <EditMessageDialog
                    message={getFullTextContent()}
                    onSave={handleEdit}
                  />
                )}

                {onFork && !isStreaming && (
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    onClick={() => onFork(message.id)}
                    title={t('chat:actions.fork')}
                  >
                    <IconGitFork size={16} />
                  </Button>
                )}

                {onDelete && !isStreaming && (
                  <DeleteMessageDialog onDelete={handleDelete} />
                )}

                {selectedModel &&
                  onContinue &&
                  !isStreaming &&
                  isLastMessage &&
                  isStopped && (
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      onClick={handleContinue}
                      title={t('chat:actions.continue')}
                    >
                      <IconPlayerPlay size={16} />
                    </Button>
                  )}

                {selectedModel && onRegenerate && !isStreaming && isLastMessage && (
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    onClick={handleRegenerate}
                    title={t('chat:actions.regenerate')}
                  >
                    <IconRefresh size={16} />
                  </Button>
                )}
                </div>
              </div>

              <TokenSpeedIndicator
                streaming={isStreaming}
                metadata={metadata}
              />
            </div>
          )}

        {/* Image Preview Dialog -- the same zoom popup the hub sheets use:
            fade+zoom animation, click outside or on the image to dismiss. The
            dialog portals itself, so the message list's top-edge fade mask
            can't clip it. */}
        {previewMounted && (
          <Dialog
            open={!!previewImage}
            onOpenChange={(open) => !open && setPreviewImage(null)}
          >
            <DialogContent
              showCloseButton={false}
              className="w-auto max-w-[90vw] max-h-[90vh] p-0 gap-0 overflow-hidden bg-transparent border-0 shadow-none"
            >
              <DialogTitle className="sr-only">
                {previewImage?.filename || 'Image'}
              </DialogTitle>
              {lastPreviewUrlRef.current && (
                <img
                  src={lastPreviewUrlRef.current}
                  alt={previewImage?.filename || 'Preview'}
                  onClick={() => setPreviewImage(null)}
                  className="block max-w-full max-h-[85vh] object-contain rounded-lg cursor-zoom-out"
                />
              )}
            </DialogContent>
          </Dialog>
        )}
      </div>
    )
  },
  (prevProps, nextProps) => {
    // Always re-render if the last message is in-flight (streaming or submitted)
    if (
      nextProps.isLastMessage &&
      (nextProps.status === CHAT_STATUS.STREAMING ||
        nextProps.status === CHAT_STATUS.SUBMITTED)
    ) {
      return false
    }

    return (
      prevProps.message === nextProps.message &&
      prevProps.isFirstMessage === nextProps.isFirstMessage &&
      prevProps.isLastMessage === nextProps.isLastMessage &&
      prevProps.status === nextProps.status &&
      prevProps.hideActions === nextProps.hideActions &&
      prevProps.charAvatarUrl === nextProps.charAvatarUrl &&
      prevProps.userAvatarUrl === nextProps.userAvatarUrl &&
      prevProps.userAvatarHidden === nextProps.userAvatarHidden &&
      prevProps.versionInfo?.index === nextProps.versionInfo?.index &&
      prevProps.versionInfo?.count === nextProps.versionInfo?.count
    )
  }
)

MessageItem.displayName = 'MessageItem'
