import type { Metadata } from 'next';

import { MarkdownDocument } from '@/components/markdown-document';

import attention from '../../../cheatsheets/attention.md?raw';

export const metadata: Metadata = {
  title: 'Attention',
  description:
    'A concise, progressive guide to self-attention, dot-product attention, scaled attention, and multi-head self-attention.',
  openGraph: {
    title: 'Attention · AI Notes',
    description:
      'A concise, progressive guide to self-attention, dot-product attention, scaled attention, and multi-head self-attention.',
    images: [],
  },
  twitter: {
    card: 'summary',
    title: 'Attention · AI Notes',
    description:
      'A concise, progressive guide to self-attention, dot-product attention, scaled attention, and multi-head self-attention.',
    images: [],
  },
};

const sections = [
  ['#1-self-attention', 'Self-attention'],
  ['#2-dot-product-self-attention', 'Dot product'],
  ['#3-scaled-dot-product-self-attention', 'Scaled dot product'],
  ['#4-multi-head-self-attention', 'Multi-head'],
] as const;

export default function AttentionCheatsheet() {
  return (
    <main className="article-shell page-shell">
      <aside className="article-sidebar">
        <span>CHEATSHEET / 001</span>
        <nav aria-label="On this page">
          {sections.map(([href, label]) => (
            <a href={href} key={href}>
              {label}
            </a>
          ))}
        </nav>
      </aside>
      <article>
        <div className="article-meta">
          Architecture · Attention · 5 min read
        </div>
        <MarkdownDocument source={attention} />
      </article>
      <aside className="article-aside">
        <strong>Reading the notation</strong>
        <p>
          Each token is a column. Rows in KᵀQ are source positions; columns are
          the positions gathering context. Softmax runs down each column.
        </p>
      </aside>
    </main>
  );
}
