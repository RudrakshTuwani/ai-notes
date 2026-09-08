import type { Metadata } from 'next';
import 'katex/dist/katex.min.css';
import './globals.css';

import { SiteHeader } from '@/components/site-header';

export const metadata: Metadata = {
  metadataBase: new URL(
    'https://rudraksh-ai-notes.rudrakshtuwani.chatgpt.site',
  ),
  title: {
    default: 'AI Notes',
    template: '%s · AI Notes',
  },
  description:
    'Concise, first-principles notes on language model architecture, systems, training, and post-training.',
  openGraph: {
    title: 'AI Notes',
    description:
      'Concise, first-principles notes on language model architecture, systems, training, and post-training.',
    images: [{ url: '/og.png', width: 1731, height: 909, alt: 'AI Notes' }],
    siteName: 'AI Notes',
    type: 'website',
    url: '/',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'AI Notes',
    description:
      'Concise, first-principles notes on language model architecture, systems, training, and post-training.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <SiteHeader />
        {children}
        <footer className="site-footer page-shell">
          <p>AI Notes · A compact field guide to language models.</p>
          <a
            href="https://github.com/RudrakshTuwani/ai-notes"
            rel="noreferrer"
            target="_blank"
          >
            View on GitHub
          </a>
        </footer>
      </body>
    </html>
  );
}
