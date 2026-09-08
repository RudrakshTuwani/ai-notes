import type { ComponentPropsWithoutRef } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

type MarkdownDocumentProps = {
  compact?: boolean;
  source: string;
};

function MarkdownLink({ href, ...props }: ComponentPropsWithoutRef<'a'>) {
  const resolvedHref =
    href?.startsWith('cheatsheets/') && href.endsWith('.md')
      ? `/${href.slice(0, -3)}`
      : href;
  const external = resolvedHref?.startsWith('http');
  return (
    <a
      href={resolvedHref}
      rel={external ? 'noreferrer' : undefined}
      target={external ? '_blank' : undefined}
      {...props}
    />
  );
}

export function MarkdownDocument({
  compact = false,
  source,
}: MarkdownDocumentProps) {
  return (
    <div className={`note-prose${compact ? ' compact' : ''}`}>
      <ReactMarkdown
        components={{ a: MarkdownLink }}
        rehypePlugins={[rehypeKatex, rehypeSlug]}
        remarkPlugins={[remarkGfm, remarkMath]}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}
