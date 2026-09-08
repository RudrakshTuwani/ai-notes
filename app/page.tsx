import { ArrowRight, BookOpen, CheckCircle2, Layers3 } from 'lucide-react';
import Link from 'next/link';

import { MarkdownDocument } from '@/components/markdown-document';

import plan from '../plan.md?raw';
import resources from '../resources.md?raw';

const topics = [
  ['01', 'Self-attention'],
  ['02', 'Dot product'],
  ['03', 'Scaled attention'],
  ['04', 'Multi-head'],
] as const;

export default function Home() {
  return (
    <main>
      <section className="hero page-shell">
        <div className="hero-copy">
          <p className="eyebrow">A living field guide to language models</p>
          <h1>
            Learn the model.
            <span>Understand the machine.</span>
          </h1>
          <p className="hero-lede">
            Compact notes on architecture, systems, training, and
            post-training—built from first principles and designed for quick
            recall.
          </p>
          <div className="hero-actions">
            <Link className="primary-link" href="/cheatsheets/attention">
              Read the attention cheatsheet
              <ArrowRight aria-hidden="true" />
            </Link>
            <a
              className="text-link"
              href="https://github.com/RudrakshTuwani/ai-notes"
              rel="noreferrer"
              target="_blank"
            >
              Browse the source
            </a>
          </div>
        </div>

        <div className="attention-map" aria-label="Attention pipeline overview">
          <div className="map-heading">
            <span>ATTN / 001</span>
            <span>FROM INPUT TO CONTEXT</span>
          </div>
          <div className="map-input">
            <span className="map-symbol">X</span>
            <span>token embeddings</span>
          </div>
          <div className="map-stem" />
          <div className="map-projections">
            <div>
              <strong>Q</strong>
              <span>ask</span>
            </div>
            <div>
              <strong>K</strong>
              <span>match</span>
            </div>
            <div>
              <strong>V</strong>
              <span>carry</span>
            </div>
          </div>
          <div className="map-equation">
            <span>softmax</span>
            <strong>KᵀQ / √d</strong>
          </div>
          <div className="map-result">
            <span>contextualized output</span>
            <strong>Y = V · A</strong>
          </div>
        </div>
      </section>

      <section
        className="page-shell feature-section"
        aria-labelledby="featured-title"
      >
        <div className="section-heading">
          <div>
            <p className="eyebrow">Featured cheatsheet</p>
            <h2 id="featured-title">Attention, progressively explained</h2>
          </div>
          <p>One notation system. Four increasingly expressive mechanisms.</p>
        </div>

        <Link className="feature-card" href="/cheatsheets/attention">
          <div className="feature-icon">
            <Layers3 aria-hidden="true" />
          </div>
          <div className="feature-content">
            <span className="feature-meta">Architecture · 5 min read</span>
            <h3>Attention</h3>
            <p>
              Build from generic weighted values to multi-head self-attention,
              with dimensions and normalization made explicit at every step.
            </p>
          </div>
          <ArrowRight className="feature-arrow" aria-hidden="true" />
          <div className="topic-strip" aria-hidden="true">
            {topics.map(([number, title]) => (
              <div key={number}>
                <span>{number}</span>
                <strong>{title}</strong>
              </div>
            ))}
          </div>
        </Link>
      </section>

      <section
        className="page-shell library-grid"
        aria-label="Learning plan and resources"
      >
        <article className="library-panel" id="plan">
          <div className="panel-label">
            <CheckCircle2 aria-hidden="true" />
            Learning path
          </div>
          <MarkdownDocument compact source={plan} />
        </article>
        <article
          className="library-panel library-panel-resources"
          id="resources"
        >
          <div className="panel-label">
            <BookOpen aria-hidden="true" />
            Reference shelf
          </div>
          <MarkdownDocument compact source={resources} />
        </article>
      </section>
    </main>
  );
}
