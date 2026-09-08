import { Code2 } from 'lucide-react';
import Link from 'next/link';

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="header-inner page-shell">
        <Link className="brand" href="/" aria-label="AI Notes home">
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
          </span>
          <span>AI Notes</span>
        </Link>
        <nav className="site-nav" aria-label="Primary navigation">
          <Link href="/cheatsheets/attention">Cheatsheets</Link>
          <Link href="/#plan">Plan</Link>
          <Link href="/#resources">Resources</Link>
          <a
            className="github-link"
            href="https://github.com/RudrakshTuwani/ai-notes"
            rel="noreferrer"
            target="_blank"
          >
            <Code2 aria-hidden="true" />
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
