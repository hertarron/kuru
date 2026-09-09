import { describe, it, expect } from 'vitest'
import { cleanModelReadme, huggingFaceRepoUrl } from '../model-readme'

describe('cleanModelReadme', () => {
  it('drops the YAML header and rewrites a banner block', () => {
    const cleaned = cleanModelReadme(
      [
        '---',
        'library_name: transformers',
        'license: apache-2.0',
        '---',
        '',
        '<div align="center">',
        '<img src=https://example.com/banner.png>',
        '<a href="https://huggingface.co/google" target="_blank">Hugging Face</a> |',
        '<a href="https://github.com/google"><b>GitHub</b></a>',
        '</div>',
        '',
        '# Gemma',
      ].join('\n')
    )

    expect(cleaned).not.toContain('library_name')
    expect(cleaned).not.toContain('<div')
    expect(cleaned).not.toContain('banner.png')
    expect(cleaned).toContain('[Hugging Face](https://huggingface.co/google)')
    expect(cleaned).toContain('[GitHub](https://github.com/google)')
    expect(cleaned).toContain('# Gemma')
  })

  it('leaves HTML inside fenced code alone', () => {
    const source = ['# Usage', '', '```html', '<div>keep me</div>', '```'].join(
      '\n'
    )
    expect(cleanModelReadme(source)).toContain('<div>keep me</div>')
  })
})

describe('huggingFaceRepoUrl', () => {
  it('derives the repo page from the README url', () => {
    expect(
      huggingFaceRepoUrl(
        'https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/resolve/main/README.md'
      )
    ).toBe('https://huggingface.co/unsloth/gemma-3-27b-it-GGUF')
  })

  it('falls back to developer and model name', () => {
    expect(huggingFaceRepoUrl(undefined, 'Menlo', 'Jan-nano-gguf')).toBe(
      'https://huggingface.co/Menlo/Jan-nano-gguf'
    )
    expect(huggingFaceRepoUrl(undefined, undefined, 'Menlo/Jan-nano')).toBe(
      'https://huggingface.co/Menlo/Jan-nano'
    )
  })
})
