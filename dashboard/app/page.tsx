'use client';

import {
  Activity,
  BrainCircuit,
  FlaskConical,
  Network,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, XAxis, YAxis } from 'recharts';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';

const methodData = [
  {
    method: 'Pearson',
    accuracy: 75.4,
    sensitivity: 77.8,
    specificity: 72.4,
    f1: 77.8,
    auc: 74.9,
  },
  {
    method: 'Spearman',
    accuracy: 76.9,
    sensitivity: 77.8,
    specificity: 75.9,
    f1: 78.9,
    auc: 74.7,
  },
  {
    method: 'Coherence',
    accuracy: 78.5,
    sensitivity: 69.4,
    specificity: 89.7,
    f1: 78.1,
    auc: 74.8,
  },
];
const folds = [
  {
    fold: 1,
    epoch: 14,
    threshold: 0.582,
    accuracy: 0.615,
    f1: 0.667,
    auc: 0.667,
  },
  {
    fold: 2,
    epoch: 1,
    threshold: 0.534,
    accuracy: 0.923,
    f1: 0.933,
    auc: 0.975,
  },
  {
    fold: 3,
    epoch: 4,
    threshold: 0.658,
    accuracy: 0.769,
    f1: 0.769,
    auc: 0.929,
  },
  {
    fold: 4,
    epoch: 15,
    threshold: 0.564,
    accuracy: 0.769,
    f1: 0.727,
    auc: 0.714,
  },
  {
    fold: 5,
    epoch: 50,
    threshold: 0.503,
    accuracy: 0.692,
    f1: 0.778,
    auc: 0.762,
  },
];
const spectral = [
  ['O2', 'Theta / alpha ratio', '3.178', '1.447', '+0.718', '6.29×10⁻⁵'],
  ['T5', 'Theta / alpha ratio', '3.376', '1.607', '+0.713', '6.29×10⁻⁵'],
  ['O2', 'Relative alpha power', '0.032', '0.080', '−0.680', '1.28×10⁻⁴'],
  ['T5', 'Relative alpha power', '0.029', '0.070', '−0.661', '1.55×10⁻⁴'],
  ['T6', 'Theta / alpha ratio', '3.308', '1.676', '+0.657', '1.55×10⁻⁴'],
  ['O2', 'Slow / fast ratio', '51.959', '26.878', '+0.653', '1.55×10⁻⁴'],
];
const connectivity = [
  ['O2', 'Weighted clustering', '0.640', '0.411', '+0.697', '4.80×10⁻⁵'],
  ['O2', 'Node strength', '4.738', '2.680', '+0.688', '4.80×10⁻⁵'],
  ['T6', 'Node strength', '4.692', '3.184', '+0.594', '5.63×10⁻⁴'],
  ['F7', 'Weighted clustering', '0.450', '0.633', '−0.582', '5.63×10⁻⁴'],
  ['F7', 'Node strength', '3.202', '5.132', '−0.546', '1.24×10⁻³'],
  ['F4', 'Node strength', '4.927', '6.483', '−0.533', '1.54×10⁻³'],
];
const chartConfig = {
  accuracy: { label: 'Accuracy', color: '#16a085' },
  sensitivity: { label: 'Sensitivity', color: '#e09f3e' },
  specificity: { label: 'Specificity', color: '#355070' },
  auc: { label: 'ROC–AUC', color: '#9b5de5' },
};
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

function EvidenceTable({ rows }: { rows: string[][] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200">
      <table className="w-full min-w-[680px] text-left text-sm">
        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            {[
              'Channel',
              'Biomarker',
              'AD mean',
              'CN mean',
              'Effect',
              'FDR q',
            ].map((x) => (
              <th key={x} className="px-4 py-3 font-semibold">
                {x}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((row, i) => (
            <tr key={`${row[0]}-${row[1]}`} className="hover:bg-teal-50/40">
              {row.map((cell, j) => (
                <td
                  key={j}
                  className={`px-4 py-3 ${j === 0 ? 'font-semibold text-slate-900' : ''} ${j === 4 ? (cell.startsWith('+') ? 'text-teal-700' : 'text-blue-700') : ''}`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-40 border-b border-white/8 bg-[#071a24]/95 text-white backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-3 lg:px-8">
          <div className="flex items-center gap-3">
            <span className="grid size-9 place-items-center rounded-xl bg-teal-400/15 text-teal-300 ring-1 ring-teal-300/25">
              <BrainCircuit className="size-5" />
            </span>
            <div>
              <p className="text-sm font-semibold">EEG–AD Research</p>
              <p className="text-[11px] text-slate-400">
                Subject-level dashboard
              </p>
            </div>
          </div>
          <nav className="hidden gap-5 text-xs text-slate-300 md:flex">
            <a href="#performance">Performance</a>
            <a href="#biomarkers">Biomarkers</a>
            <a href="#validation">Validation</a>
            <a href="#methodology">Methodology</a>
          </nav>
          <span className="rounded-full bg-emerald-400/10 px-3 py-1.5 text-xs font-medium text-emerald-300 ring-1 ring-emerald-300/20">
            Analysis complete
          </span>
        </div>
      </header>

      <section className="border-b bg-[#0b2531] text-white">
        <div className="mx-auto max-w-7xl px-5 py-12 lg:px-8">
          <div className="max-w-3xl">
            <p className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[.18em] text-teal-300">
              <FlaskConical className="size-4" />
              Graph-based EEG analysis
            </p>
            <h1 className="text-3xl font-semibold leading-tight tracking-tight sm:text-5xl">
              Alzheimer’s classification & biomarker evidence
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
              A consolidated, reproducible view of GCN performance, connectivity
              experiments, qEEG biomarkers and subject-level statistical
              evidence.
            </p>
          </div>
          <div className="mt-9 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ['Subjects', '65', '36 AD · 29 CN', BrainCircuit],
              ['EEG graphs', '12,826', '19 channels · 4 bands', Network],
              ['Best accuracy', '78.5%', 'Coherence GCN', Activity],
              [
                'Pearson OOF AUC',
                '0.749',
                '65 unseen predictions',
                ShieldCheck,
              ],
            ].map(([label, value, detail, Icon]) => (
              <div
                key={String(label)}
                className="rounded-2xl border border-white/10 bg-white/[.055] p-5"
              >
                <div className="flex items-center justify-between text-slate-400">
                  <span className="text-xs font-medium uppercase tracking-wider">
                    {String(label)}
                  </span>
                  {typeof Icon !== 'string' && (
                    <Icon className="size-4 text-teal-300" />
                  )}
                </div>
                <div className="mt-3 text-3xl font-semibold tabular-nums">
                  {String(value)}
                </div>
                <p className="mt-1 text-sm text-slate-400">{String(detail)}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-7xl space-y-10 px-5 py-10 lg:px-8">
        <section id="performance" className="scroll-mt-24">
          <div className="mb-5">
            <p className="eyebrow">Model performance</p>
            <h2>Connectivity comparison</h2>
            <p className="section-copy">
              Every method uses identical subject folds, random seeds, node
              features and GCN hyperparameters.
            </p>
          </div>
          <div className="grid gap-6 lg:grid-cols-[1.7fr_1fr]">
            <Card className="border-0 shadow-sm ring-slate-200">
              <CardHeader>
                <CardTitle>Out-of-fold performance</CardTitle>
                <CardDescription>Scores shown as percentages</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={chartConfig}
                  className="h-[340px] w-full aspect-auto"
                >
                  <BarChart data={methodData}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="method" tickLine={false} axisLine={false} />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v) => `${v}%`}
                      tickLine={false}
                      axisLine={false}
                    />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Legend />
                    <Bar
                      dataKey="accuracy"
                      fill="var(--color-accuracy)"
                      radius={[4, 4, 0, 0]}
                    />
                    <Bar
                      dataKey="sensitivity"
                      fill="var(--color-sensitivity)"
                      radius={[4, 4, 0, 0]}
                    />
                    <Bar
                      dataKey="specificity"
                      fill="var(--color-specificity)"
                      radius={[4, 4, 0, 0]}
                    />
                    <Bar
                      dataKey="auc"
                      fill="var(--color-auc)"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>
            <Card className="border-0 bg-teal-950 text-white shadow-sm ring-teal-900">
              <CardHeader>
                <CardTitle className="text-xl">Interpretation</CardTitle>
                <CardDescription className="text-teal-100/65">
                  What the comparison supports
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5 leading-6 text-teal-50/85">
                <p>
                  Ranking performance is nearly unchanged: pooled ROC–AUC is{' '}
                  <strong>0.747–0.749</strong> for all methods.
                </p>
                <p>
                  Coherence increases specificity to <strong>89.7%</strong>, but
                  reduces AD sensitivity to <strong>69.4%</strong>.
                </p>
                <div className="rounded-xl bg-white/8 p-4 text-sm ring-1 ring-white/10">
                  <strong className="text-teal-200">Conclusion:</strong> the
                  higher coherence accuracy does not demonstrate a generally
                  superior classifier. Spectral node features appear to carry
                  the more stable signal.
                </div>
              </CardContent>
            </Card>
          </div>
          <div className="mt-6 grid gap-5 md:grid-cols-3">
            <figure className="figure-card">
              <img
                src="/results/oof_roc_curve.png"
                alt="Pearson out-of-fold ROC curve"
              />
              <figcaption>
                Subject-level ROC curve · Pearson reference
              </figcaption>
            </figure>
            <figure className="figure-card">
              <img
                src="/results/oof_confusion_matrix.png"
                alt="Pearson confusion matrix"
              />
              <figcaption>21 true CN, 28 true AD, 16 total errors</figcaption>
            </figure>
            <figure className="figure-card">
              <img
                src="/results/oof_probability_distribution.png"
                alt="Predicted AD probabilities by group"
              />
              <figcaption>Out-of-fold AD probability distributions</figcaption>
            </figure>
          </div>
        </section>

        <section id="biomarkers" className="scroll-mt-24">
          <div className="mb-5">
            <p className="eyebrow">Biomarker evidence</p>
            <h2>Subject-level group differences</h2>
            <p className="section-copy">
              Each patient contributes one aggregated value. Tests use two-sided
              Mann–Whitney U, rank-biserial effect sizes and Benjamini–Hochberg
              FDR correction.
            </p>
          </div>
          <div className="space-y-8">
            <div>
              <h3 className="mb-4 text-lg font-semibold text-slate-900">Spectral qEEG</h3>
              <div className="grid gap-5 lg:grid-cols-[1.45fr_1fr]">
                <Card>
                  <CardHeader>
                    <CardTitle>Strongest spectral findings</CardTitle>
                    <CardDescription>
                      78 of 133 channel–feature tests significant after FDR
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <EvidenceTable rows={spectral} />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Spatial pattern</CardTitle>
                    <CardDescription>
                      Alpha power is lower in AD, strongest posteriorly
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img
                      className="mx-auto max-h-[430px] object-contain"
                      src="/results/topomap_alpha.png"
                      alt="Scalp map of relative alpha power effect sizes"
                    />
                  </CardContent>
                </Card>
              </div>
              <div className="mt-5 grid gap-5 lg:grid-cols-2">
                <figure className="figure-card">
                  <img
                    src="/results/topomap_theta_alpha.png"
                    alt="Theta alpha ratio scalp map"
                  />
                  <figcaption>
                    Theta/alpha ratio is consistently higher in AD
                  </figcaption>
                </figure>
                <figure className="figure-card">
                  <img
                    src="/results/heatmap_theta_alpha.png"
                    alt="Theta alpha ratio channel heatmap"
                  />
                  <figcaption>
                    Positive effects across all 19 channels; strongest at O2 and
                    T5
                  </figcaption>
                </figure>
              </div>
            </div>
            <div>
              <h3 className="mb-4 text-lg font-semibold text-slate-900">Connectivity biomarkers</h3>
              <div className="grid gap-5 lg:grid-cols-[1.45fr_1fr]">
                <Card>
                  <CardHeader>
                    <CardTitle>Strongest connectivity findings</CardTitle>
                    <CardDescription>
                      23 of 43 tests significant after FDR
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <EvidenceTable rows={connectivity} />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Regional, not global</CardTitle>
                    <CardDescription>
                      Node strength effect by channel
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img
                      src="/results/connectivity_node_strength.png"
                      alt="Channel-wise connectivity node strength effects"
                    />
                  </CardContent>
                </Card>
              </div>
              <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm leading-6 text-amber-950">
                <div className="flex gap-3">
                  <AlertTriangle className="mt-0.5 size-5 shrink-0 text-amber-600" />
                  <p>
                    <strong>Interpret carefully.</strong> The Pearson
                    connectivity biomarker analysis uses absolute, top-30%
                    thresholded edge weights. It detects regional
                    magnitude/topology differences but cannot distinguish
                    positive from negative correlations in the original
                    reference dataset.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="validation" className="scroll-mt-24">
          <div className="mb-5">
            <p className="eyebrow">Cross-validation detail</p>
            <h2>Pearson reference by fold</h2>
            <p className="section-copy">
              Thresholds were selected only from each validation fold using
              Youden’s J and applied once to held-out test subjects.
            </p>
          </div>
          <Card>
            <CardContent className="pt-1">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-sm">
                  <thead>
                    <tr className="border-b text-left text-xs uppercase tracking-wide text-slate-500">
                      {[
                        'Fold',
                        'Best epoch',
                        'Threshold',
                        'Accuracy',
                        'F1',
                        'ROC–AUC',
                        'Test subjects',
                      ].map((x) => (
                        <th key={x} className="px-4 py-3">
                          {x}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {folds.map((x) => (
                      <tr
                        key={x.fold}
                        className="border-b border-slate-100 last:border-0"
                      >
                        <td className="px-4 py-3 font-semibold">{x.fold}</td>
                        <td className="px-4 py-3">{x.epoch}</td>
                        <td className="px-4 py-3 font-mono">
                          {x.threshold.toFixed(3)}
                        </td>
                        <td className="px-4 py-3">{pct(x.accuracy)}</td>
                        <td className="px-4 py-3">{x.f1.toFixed(3)}</td>
                        <td className="px-4 py-3">{x.auc.toFixed(3)}</td>
                        <td className="px-4 py-3">13</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
          <div className="mt-5 grid gap-4 sm:grid-cols-3">
            <div className="mini-callout">
              <span>Fold accuracy</span>
              <strong>0.754 ± 0.102</strong>
            </div>
            <div className="mini-callout">
              <span>Fold F1</span>
              <strong>0.775 ± 0.088</strong>
            </div>
            <div className="mini-callout">
              <span>Fold ROC–AUC</span>
              <strong>0.809 ± 0.121</strong>
            </div>
          </div>
        </section>

        <section id="methodology" className="scroll-mt-24">
          <div className="mb-5">
            <p className="eyebrow">Methodology</p>
            <h2>From EEG recording to evidence</h2>
          </div>
          <div className="grid gap-4 md:grid-cols-5">
            {[
              [
                '01',
                'Preprocess',
                '0.5–45 Hz, average reference, 4-second epochs',
              ],
              [
                '02',
                'Node features',
                'Relative delta, theta, alpha and beta power',
              ],
              [
                '03',
                'Graph edges',
                'Top 30% Pearson, Spearman or coherence links',
              ],
              [
                '04',
                'Subject GCN',
                'Mean AD probability across all patient epochs',
              ],
              [
                '05',
                'Evaluation',
                '5-fold subject-wise CV with no patient leakage',
              ],
            ].map(([n, t, d]) => (
              <div className="step-card" key={n}>
                <span>{n}</span>
                <h3>{t}</h3>
                <p>{d}</p>
              </div>
            ))}
          </div>
          <div className="mt-6 grid gap-5 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="size-5 text-teal-600" />
                  Safeguards implemented
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="clean-list">
                  <li>All epochs from one patient remain in one split</li>
                  <li>Subject-balanced training loss</li>
                  <li>Early stopping on validation loss</li>
                  <li>Decision threshold selected without test data</li>
                  <li>Every subject receives exactly one OOF prediction</li>
                  <li>FDR correction for biomarker multiplicity</li>
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="size-5 text-amber-600" />
                  Limitations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="clean-list">
                  <li>Small single-dataset sample: 65 subjects</li>
                  <li>No external cohort validation</li>
                  <li>Ratios can be unstable when alpha power is very small</li>
                  <li>Connectivity threshold fixed at top 30%</li>
                  <li>
                    Current results are exploratory, not a clinical diagnostic
                    system
                  </li>
                  <li>
                    Baseline model still requires matched 5-fold evaluation
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="rounded-3xl bg-[#0b2531] p-7 text-white sm:p-10">
          <p className="eyebrow text-teal-300">Overall conclusion</p>
          <h2 className="max-w-3xl text-white">
            A reproducible signal, with clear limits
          </h2>
          <p className="mt-4 max-w-4xl text-base leading-7 text-slate-300">
            The subject-level GCN separates AD and CN above chance, while
            posterior alpha reduction and elevated theta/alpha ratios provide
            the clearest qEEG evidence. Connectivity differences are regionally
            informative, particularly around O2, T5/T6 and F7, but changing edge
            construction does not materially improve ROC–AUC. These results
            justify further validation, not clinical deployment.
          </p>
        </section>
      </div>
      <footer className="border-t bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-5 py-6 text-xs text-slate-500 sm:flex-row sm:justify-between lg:px-8">
          <span>Graph-Based EEG Analysis for Alzheimer’s Disease</span>
          <span>
            Generated from verified project outputs · 65 subject-level
            observations
          </span>
        </div>
      </footer>
    </main>
  );
}
